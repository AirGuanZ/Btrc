#include <btrc/core/film/film.h>

BTRC_CORE_BEGIN

Film::Film()
    : width_(0), height_(0)
{
    
}

Film::Film(int width, int height)
    : width_(width), height_(height)
{
    
}

Film::Film(Film &&other) noexcept
    : Film()
{
    swap(other);
}

Film &Film::operator=(Film &&other) noexcept
{
    swap(other);
    return *this;
}

void Film::swap(Film &other) noexcept
{
    std::swap(width_, other.width_);
    std::swap(height_, other.height_);
    buffers_.swap(other.buffers_);
}

Film::operator bool() const
{
    return width_ > 0;
}

int Film::width() const
{
    return width_;
}

int Film::height() const
{
    return height_;
}

void Film::add_output(std::string name, Format format)
{
    assert(*this);
    assert(!buffers_.contains(name));

    size_t count = width_ * height_;
    if(format == Float)
        count *= 1;
    else
    {
        assert(format == Float3);
        count *= 4;
    }
    CUDABuffer<float> buffer(count);
    buffer.clear_bytes(0);
    buffers_.insert(
        { std::move(name), FilmBuffer{ format, std::move(buffer) } });
}

bool Film::has_output(std::string_view name) const
{
    return buffers_.contains(name);
}

void Film::splat(
    const CVec2f &pixel_coord,
    std::span<std::pair<std::string_view, CValue>> values)
{
    for(auto &p : values)
        splat(pixel_coord, p.first, p.second);
}

void Film::splat_atomic(
    const CVec2f &pixel_coord,
    std::span<std::pair<std::string_view, CValue>> values)
{
    for(auto &p : values)
        splat_atomic(pixel_coord, p.first, p.second);
}

void Film::splat(
    const CVec2f &pixel_coord, std::string_view name, const CValue &value)
{
    auto buffer_it = buffers_.find(name);
    if(buffer_it == buffers_.end())
        return;
    auto &buffer = buffer_it->second;

    var ptr_buffer = cuj::import_pointer(buffer.buffer.get());

    i32 xi = i32(cstd::floor(pixel_coord.x));
    i32 yi = i32(cstd::floor(pixel_coord.y));
    $if(0 <= xi & xi < width_ & 0 <= yi & yi < height_)
    {
        if(buffer.format == Float)
        {
            auto &val = value.as<f32>();
            $if(cstd::isfinite(val))
            {
                ptr_buffer[yi * width_ + xi] = ptr_buffer[yi * width_ + xi] + val;
            };
        }
        else
        {
            auto &val = value.as<CVec3f>();
            $if(isfinite(val))
            {
                ptr_buffer[(yi * width_ + xi) * 4 + 0] = ptr_buffer[(yi * width_ + xi) * 4 + 0] + val.x;
                ptr_buffer[(yi * width_ + xi) * 4 + 1] = ptr_buffer[(yi * width_ + xi) * 4 + 1] + val.y;
                ptr_buffer[(yi * width_ + xi) * 4 + 2] = ptr_buffer[(yi * width_ + xi) * 4 + 2] + val.z;
            };
        }
    };
}

void Film::splat_atomic(
    const CVec2f &pixel_coord, std::string_view name, const CValue &value)
{
    auto buffer_it = buffers_.find(name);
    if(buffer_it == buffers_.end())
        return;
    auto &buffer = buffer_it->second;

    var ptr_u64 = reinterpret_cast<uint64_t>(buffer.buffer.get());
    var ptr_buffer = cuj::bitcast<ptr<f32>>(ptr_u64);

    i32 xi = i32(cstd::floor(pixel_coord.x));
    i32 yi = i32(cstd::floor(pixel_coord.y));
    $if(0 <= xi & xi < width_ & 0 <= yi & yi < height_)
    {
        if(buffer.format == Float)
        {
            var val = value.as<f32>();
            $if(cstd::isfinite(val))
            {
                cstd::atomic_add(ptr_buffer[yi * width_ + xi].address(), val);
            };
        }
        else
        {
            var val = value.as<CVec3f>();
            $if(isfinite(val))
            {
                cstd::atomic_add(ptr_buffer[(yi * width_ + xi) * 4 + 0].address(), val.x);
                cstd::atomic_add(ptr_buffer[(yi * width_ + xi) * 4 + 1].address(), val.y);
                cstd::atomic_add(ptr_buffer[(yi * width_ + xi) * 4 + 2].address(), val.z);
            };
        }
    };
}

void Film::clear_output(std::string_view name)
{
    auto it = buffers_.find(name);
    if(it == buffers_.end())
        throw BtrcException("unknown film output name: " + std::string(name));
    it->second.buffer.clear_bytes(0);
}

const CUDABuffer<float> &Film::get_float_output(std::string_view name) const
{
    auto it = buffers_.find(name);
    if(it == buffers_.end())
        throw BtrcException("unknown film output name: " + std::string(name));
    if(it->second.format != Float)
        throw BtrcException("Film::get_float_output is called with the wrong format");
    return it->second.buffer;
}

const CUDABuffer<float> &Film::get_float3_output(std::string_view name) const
{
    auto it = buffers_.find(name);
    if(it == buffers_.end())
        throw BtrcException("unknown film output name: " + std::string(name));
    if(it->second.format != Float3)
        throw BtrcException("Film::get_float_output is called with the wrong format");
    return it->second.buffer;
}

BTRC_CORE_END
