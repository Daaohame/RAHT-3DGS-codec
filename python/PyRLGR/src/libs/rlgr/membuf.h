#ifndef MEMBUF_H
#define MEMBUF_H

#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <vector>

// --- Declare macros ---
#define LEAST8_BITS         (sizeof(uint_least8_t)*8)
#define MASK(k)             ((((uint64_t) 0x1)<<(k))-1)
#ifndef UINT_LEAST8_MAX
# define UINT_LEAST8_MAX    MASK(LEAST8_BITS)
#endif
#ifndef UINT64_MAX
# define UINT64_MAX        ((uint64_t) -1)
#endif
#define L   4
#define U0  3
#define D0  1
#define U1  2
#define D1  1
// --- End of macros ---

class membuf
{
private:
    uint64_t        data;
    uint_least8_t   bits;
    uint_least8_t   flagWrite;
    
    // --- Replaced FILE* with vector and position ---
    std::vector<uint8_t> buffer;
    size_t          pos;
    // ---
    
    void            flush();
    void            fill();

public:
    // Constructor for writing (creates an empty internal buffer)
    membuf();
    // Constructor for reading (copies the input buffer)
    membuf(const std::vector<uint8_t> &in_buf);
    void close();

    uint64_t buffer_size() {return this->buffer.size();}
    std::vector<uint8_t> get_buffer();

    uint_least8_t   eof();

    uint_least8_t   read();
    uint64_t        read(uint_least8_t bits);
    void            read(void *ptr, size_t size, size_t count);

    void            write(uint_least8_t data);
    void            write(uint64_t data, uint_least8_t bits);
    void            write(void *ptr, size_t size, size_t count);

    uint64_t        grRead(uint_least8_t bits);
    void            grWrite(uint64_t data, uint_least8_t bits);

    std::chrono::nanoseconds rlgrRead(int64_t *seq, size_t N, uint_least8_t flagSigned=1);
    std::chrono::nanoseconds rlgrWrite(int64_t *seq, size_t N, uint_least8_t flagSigned=1);
};

uint64_t _s2u(int64_t val);
int64_t _u2s(uint64_t val);

#endif // MEMBUF_H