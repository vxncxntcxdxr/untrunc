#include "mp4.h"
#include "file.h"
#include "hvc1/nal.h"

#include <cstring>
#include <limits>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/mem.h>
}

using namespace std;

namespace {

uint32_t readLe32(const uchar* p) {
	return uint32_t(p[0]) | uint32_t(p[1]) << 8 |
	       uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

struct CanonHeader {
	string type;
	uint32_t length = 0, sample_size = 0, n_samples = 0;
	vector<uint32_t> sizes;

	uint32_t sizeAt(uint32_t i) const {
		return sample_size ? sample_size : sizes[i];
	}
};

// CAEP recordings can store a little-endian recovery index before each group
// of samples: "repi", 24-bit header length, 0x83, type, fixed size, sample count,
// then one size per sample when the fixed size is zero.
bool readCanonHeader(BufferedAtom& mdat, off_t off, CanonHeader& header) {
	if (off < 0 || mdat.contentSize() - off < 20) return false;
	auto p = mdat.getFragment(off, 20);
	if (memcmp(p, "repi", 4) || p[7] != 0x83) return false;

	CanonHeader h;
	h.type = string(reinterpret_cast<const char*>(p + 8), 4);
	if (!contains({"vide", "aac ", "tmcd"}, h.type)) return false;
	h.length = readLe32(p + 4) & 0xffffff;
	h.sample_size = readLe32(p + 12);
	h.n_samples = readLe32(p + 16);
	if (!h.n_samples || h.sample_size > uint32_t(numeric_limits<int>::max())) return false;
	uint64_t expected = 20 + (h.sample_size ? 0 : uint64_t(h.n_samples) * 4);
	if (h.length != expected || h.length > mdat.contentSize() - off) return false;
	// Even an incomplete payload must contain at least one complete sample.
	if (h.sample_size && h.sample_size > mdat.contentSize() - off - h.length) return false;

	if (!h.sample_size) {
		h.sizes.reserve(h.n_samples);
		for (uint32_t i = 0; i < h.n_samples; ++i) {
			auto size = readLe32(mdat.getFragment(off + 20 + uint64_t(i) * 4, 4));
			if (!size || size > uint32_t(numeric_limits<int>::max())) return false;
			h.sizes.push_back(size);
		}
	}
	header = move(h);
	return true;
}

int canonTrack(const vector<Track>& tracks, const string& type) {
	int found = -1;
	for (size_t i = 0; i < tracks.size(); ++i) {
		auto& t = tracks[i];
		bool matches = (type == "vide" && contains({"hvc1", "hev1"}, t.codec_.name_)) ||
		               (type == "aac " && t.codec_.name_ == "mp4a") ||
		               (type == "tmcd" && t.codec_.name_ == "tmcd");
		if (!matches) continue;
		if (found >= 0) return -1;  // The index does not identify duplicate tracks.
		found = i;
	}
	return found;
}

// Search only between indexed payloads. A bare "repi" inside compressed data
// is insufficient: the complete header, type and size table must also validate.
off_t findCanonHeader(BufferedAtom& mdat, off_t off, CanonHeader& header) {
	const int window = 1 << 16;
	while (mdat.contentSize() - off >= 20) {
		int size = min<int64_t>(window, mdat.contentSize() - off);
		auto p = mdat.getFragment(off, size);
		int candidate = -1;
		for (int i = 0; i <= size - 20; ++i) {
			if (!memcmp(p + i, "repi", 4) && p[i + 7] == 0x83) {
				candidate = i;
				break;
			}
		}
		if (candidate >= 0) {
			off += candidate;
			if (readCanonHeader(mdat, off, header)) return off;
			++off;  // Reload the search buffer after reading a rejected header.
		}
		else off += size - 19;
	}
	return -1;
}

void freeCodecContext(AVCodecContext* context) {
	avcodec_free_context(&context);
}

struct CanonPicture {
	int poc;
	bool keyframe, resets_poc;
};

class CanonHevcParser {
	unique_ptr<AVCodecContext, decltype(&freeCodecContext)> context_{nullptr, freeCodecContext};
	unique_ptr<AVCodecParserContext, decltype(&av_parser_close)> parser_{nullptr, av_parser_close};
public:
	explicit CanonHevcParser(const AVCodecParameters* parameters) {
		context_.reset(avcodec_alloc_context3(nullptr));
		parser_.reset(av_parser_init(AV_CODEC_ID_HEVC));
		if (!context_ || !parser_)
			throw string("Canon recovery requires the FFmpeg HEVC parser (--enable-parser=hevc)");
		if (avcodec_parameters_to_context(context_.get(), parameters) < 0)
			throw string("Could not initialize Canon HEVC parameters");
		parser_->flags |= PARSER_FLAG_COMPLETE_FRAMES;

		// FFmpeg 3.x's parser expects Annex B, including its parameter sets.
		auto p = parameters->extradata;
		int length = parameters->extradata_size;
		if (length < 23 || p[0] != 1 || (p[21] & 3) != 3)
			throw string("Unsupported Canon HEVC configuration");
		vector<uint8_t> annexb;
		int off = 23;
		for (int array = 0; array < p[22]; ++array) {
			if (length - off < 3) throw string("Incomplete Canon HEVC configuration");
			int count = (p[off + 1] << 8) | p[off + 2];
			off += 3;
			for (int i = 0; i < count; ++i) {
				if (length - off < 2) throw string("Incomplete Canon HEVC configuration");
				int size = (p[off] << 8) | p[off + 1];
				off += 2;
				if (size > length - off) throw string("Incomplete Canon HEVC configuration");
				annexb.insert(annexb.end(), {0, 0, 0, 1});
				annexb.insert(annexb.end(), p + off, p + off + size);
				off += size;
			}
		}
		av_freep(&context_->extradata);
		context_->extradata_size = annexb.size();
		context_->extradata = static_cast<uint8_t*>(av_mallocz(annexb.size() + AV_INPUT_BUFFER_PADDING_SIZE));
		if (!context_->extradata) throw string("Could not allocate Canon HEVC configuration");
		memcpy(context_->extradata, annexb.data(), annexb.size());
	}

	CanonPicture parse(BufferedAtom& mdat, off_t off, uint32_t size) {
		while (size >= 6) {
			H265NalInfo nal(mdat.getFragment(off, 6), size);
			if (!nal.is_ok || nal.length_ < 6) break;
			if (nal.nal_type_ <= 31) {
				// Picture order is in the first slice header; compressed pixels and
				// subsequent slices are unnecessary for the parser.
				int count = min<uint32_t>(nal.length_ - 4, 64);
				vector<uint8_t> annexb{0, 0, 0, 1};
				auto data = mdat.getFragment(off + 4, count);
				annexb.insert(annexb.end(), data, data + count);
				int length = annexb.size();
				annexb.resize(length + AV_INPUT_BUFFER_PADDING_SIZE);
				uint8_t* out;
				int out_size;
				parser_->output_picture_number = numeric_limits<int>::min();
				av_parser_parse2(parser_.get(), context_.get(), &out, &out_size,
				                 annexb.data(), length, AV_NOPTS_VALUE, AV_NOPTS_VALUE, -1);
				if (parser_->output_picture_number == numeric_limits<int>::min()) break;
				return {parser_->output_picture_number, h265IsKeyframe(nal.nal_type_),
				        nal.nal_type_ >= 16 && nal.nal_type_ <= 20};
			}
			off += nal.length_;
			size -= nal.length_;
		}
		throw ss("Could not parse indexed Canon HEVC picture at ", g_mp4->offToStr(off));
	}
};

} // namespace

bool Mp4::repairCanon() {
	CanonHeader header;
	auto& mdat = *current_mdat_;
	if (!has_canon_brand_ || !readCanonHeader(mdat, 0, header) ||
	    canonTrack(tracks_, header.type) < 0) return false;
	int video_idx = canonTrack(tracks_, "vide");
	if (video_idx < 0) return false;
	CanonHevcParser parser(tracks_[video_idx].codec_.av_codec_params_);
	vector<pair<int, size_t>> group; // (picture order count, decode index)
	vector<size_t> presentation_order;
	auto finishGroup = [&]() {
		sort(group.begin(), group.end());
		for (auto picture : group) presentation_order.push_back(picture.second);
		group.clear();
	};

	logg(I, "using Canon repi recovery indexes\n");
	off_t offset = 0;
	while (offset < mdat.contentSize()) {
		auto header_off = findCanonHeader(mdat, offset, header);
		if (header_off < 0) {
			addToExclude(offset, mdat.contentSize() - offset);
			break;
		}
		int idx = canonTrack(tracks_, header.type);
		if (idx < 0) throw string("Canon recovery index has no unique matching reference track");
		auto& track = tracks_[idx];
		if (header.sample_size && track.constant_size_ != int(header.sample_size))
			throw string("Canon recovery index sample size does not match the reference track");

		auto payload = header_off + header.length;
		addToExclude(offset, payload - offset);
		offset = payload;
		logg(V, "Canon ", header.type, " index at ", offToStr(header_off),
		     ": ", header.n_samples, " samples\n");

		for (uint32_t i = 0; i < header.n_samples; ++i) {
			uint32_t size = header.sizeAt(i);
			if (track.constant_size_ && size != uint32_t(track.constant_size_))
				throw string("Canon recovery index sample size does not match the reference track");
			if (size > mdat.contentSize() - offset) {
				logg(W, "incomplete Canon sample at ", offToStr(offset), "\n");
				addToExclude(offset, mdat.contentSize() - offset);
				offset = mdat.contentSize();
				break;
			}
			bool keyframe = false;
			if (header.type == "vide") {
				if (size < 7) throw string("Canon video sample is too short");
				auto picture = parser.parse(mdat, offset, size);
				if (picture.resets_poc) finishGroup();
				group.emplace_back(picture.poc, track.getNumSamples());
				keyframe = picture.keyframe;
			}

			FrameInfo frame(idx, keyframe, 0, offset, size);
			if (use_offset_map_ && !track.constant_size_) chkFrameDetectionAt(frame, offset);
			Track::Chunk chunk(offset, size, 1);
			chunk.already_excluded_ = mdat.total_excluded_yet_;
			track.chunks_.push_back(chunk);
			addFrame(frame);
			++pkt_idx_;
			offset += size;

			int64_t padding = (track.pkt_sz_gcd_ - size % track.pkt_sz_gcd_) % track.pkt_sz_gcd_;
			padding = min<int64_t>(padding, mdat.contentSize() - offset);
			if (padding) addToExclude(offset, padding);
			offset += padding;
		}
	}
	finishGroup();
	auto& video = tracks_[video_idx];
	video.fixTimes();
	vector<int64_t> presentation_times(presentation_order.size());
	int64_t time = 0;
	for (auto idx : presentation_order) {
		presentation_times[idx] = time;
		time += video.getTime(idx);
	}
	time = 0;
	for (size_t i = 0; i < presentation_times.size(); ++i) {
		auto offset = presentation_times[i] - time;
		if (offset < numeric_limits<int>::min() || offset > numeric_limits<int>::max())
			throw string("Canon composition offset is too large");
		video.composition_offsets_.push_back(offset);
		time += video.getTime(i);
	}
	if (!presentation_times.empty() && !g_no_ctts) {
		// An explicit edit keeps leading pictures at time zero even though the
		// first packet in decode order can have a later presentation timestamp.
		auto edts = new Atom;
		edts->name_ = "edts";
		auto elst = new Atom;
		elst->name_ = "elst";
		auto duration = video.getDurationInTimescale();
		bool wide = duration > numeric_limits<uint32_t>::max();
		elst->content_.resize(wide ? 28 : 20);
		elst->writeInt(wide ? 0x01000000 : 0, 0);
		elst->writeInt(1, 4);
		if (wide) {
			elst->writeInt64(duration, 8);
			elst->writeInt64(0, 16);
		}
		else {
			elst->writeInt(duration, 8);
			elst->writeInt(0, 12);
		}
		elst->writeInt(0x00010000, wide ? 24 : 16);
		edts->children_.push_back(elst);
		video.trak_->children_.push_back(edts);
	}
	return true;
}
