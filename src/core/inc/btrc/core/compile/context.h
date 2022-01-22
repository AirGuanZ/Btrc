#pragma once

#include <map>
#include <vector>

#include <cuj.h>

#include <btrc/core/utils/uncopyable.h>

BTRC_CORE_BEGIN

class CompileContext;

template<typename T>
class AttributeSlot : public Uncopyable
{
public:

    AttributeSlot();

    AttributeSlot(CompileContext *context, void *device_ptr, size_t count);

    AttributeSlot(AttributeSlot &&other) noexcept;

    AttributeSlot &operator=(AttributeSlot &&other) noexcept;

    ~AttributeSlot();

    void swap(AttributeSlot &other) noexcept;

    operator bool() const;

    void set(const T *cpu_data);

    cuj::cxx<T> get() const;

private:

    CompileContext *context_;
    void           *device_ptr_;
    size_t          count_;
};

class CompileContext : public Uncopyable
{
public:

    CompileContext();

    template<typename T>
    AttributeBuffer<T> allocate(size_t count);

    void _free(void *ptr, size_t bytes);

private:

    std::map<size_t, std::vector<void *>> free_attributes_;
};

BTRC_CORE_END
