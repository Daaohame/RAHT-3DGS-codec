#include "membuf.h"
#include <chrono>

uint64_t _s2u(int64_t val)
{
    uint64_t dval;
    if( val<0 ) {
        dval = -val;
        return (dval<<1)-1;
    } else {
        dval = val;
        return (dval<<1);
    }
}

int64_t _u2s(uint64_t val)
{
    int64_t dval = val>>1;
    if( val&0x1 )
        return -dval-1;
    else
        return dval;
}

// Write constructor
membuf::membuf()
{
    this->flagWrite = 1; // Write mode
    this->pos = 0;
    this->bits = 0;
    this->data = 0;
    // this->buffer is automatically created as an empty vector
}

// Read constructor
membuf::membuf(const std::vector<uint8_t> &in_buf)
    : buffer(in_buf) // <-- Copies the input vector into our internal one
{
    this->flagWrite = 0; // Read mode
    this->pos = 0;
    this->bits = 0;
    this->data = 0;
}

// Close: In write mode, flushes any remaining bits
void membuf::close()
{
    if( this->flagWrite )
    {
        uint_least8_t r = this->bits%LEAST8_BITS;
        if( r )
            this->write(0, LEAST8_BITS-r); // Pad to full byte
        else
            this->flush(); // Flush remaining full bytes
    }
}

// Returns a copy of the internal buffer
std::vector<uint8_t> membuf::get_buffer()
{
    return this->buffer; 
}

// eof: Checks if we are at the end of the vector
uint_least8_t membuf::eof()
{
    this->fill();
    return  (this->pos >= this->buffer.size()) && // At end of buffer
            (!(this->bits/LEAST8_BITS)) &&
            (!(this->data&MASK(this->bits)));
}

// flush: Writes buffered bits into the vector
void membuf::flush()
{
    uint8_t byte;
    while( this->bits>=LEAST8_BITS )
    {
        this->bits -= LEAST8_BITS;
        byte = ((this->data)>>(this->bits)) & UINT_LEAST8_MAX;
        
        // Write the byte to the vector
        this->buffer.push_back(byte); 
        this->pos++; // Increment position
    }
}

// fill: Reads bytes from the vector into the bit buffer
void membuf::fill()
{
    uint8_t byte;
    while( this->bits<=(64-LEAST8_BITS) )
    {
        // Check if we are still within the buffer bounds
        if( this->pos < this->buffer.size() )
        {
            byte = this->buffer[this->pos];
            this->pos++;
            this->data = (this->data<<LEAST8_BITS) + byte;
            this->bits += LEAST8_BITS;
        }
        else
            return; // No more data to read
    }
}

uint_least8_t membuf::read()
{
    if( !this->bits )
        this->fill();
    this->bits--;
    return (this->data>>this->bits)&0x1;
}

uint64_t membuf::read(uint_least8_t bits)
{
    if( bits>(64-LEAST8_BITS) )
    {
        uint64_t data = this->read(bits-64/2)<<(64/2);
        return  data + this->read(64/2);
    }

    this->fill();
    this->bits -= bits;
    return  (this->data>>this->bits) & MASK(bits);
}


void membuf::read(void *ptr, size_t size, size_t count)
{
    if( size==8 )
    {
        while( count-- )
        {
            *((uint64_t *) ptr) = this->read( 64 );
            ptr = ((uint64_t *) ptr) + 1;
        }
        return;
    }
    if( size==4 )
    {
        while( count-- )
        {
            *((uint32_t *) ptr) = this->read( 32 );
            ptr = ((uint32_t *) ptr) + 1;
        }
        return;
    }
    if( size==2 )
    {
        while( count-- )
        {
            *((uint16_t *) ptr) = this->read( 16 );
            ptr = ((uint16_t *) ptr) + 1;
        }
        return;
    }
    count *= size;
    while( count-- )
    {
        *((uint8_t *) ptr) = this->read( 8 );
        ptr = ((uint8_t *) ptr) + 1;
    }
}

void membuf::write(uint_least8_t data)
{
    this->data <<= 1;
    if( data )
        this->data++;
    this->bits++;

    if( this->bits>=LEAST8_BITS )
        this->flush();
}

void membuf::write(uint64_t data, uint_least8_t bits)
{
    if( bits>(64-LEAST8_BITS) )
    {
        this->write(data>>(64/2), bits-64/2);
        this->write(data&MASK(64/2), 64/2);
        return;
    }

    this->data = (this->data<<bits) + data;
    this->bits += bits;
    this->flush();
}

void membuf::write(void *ptr, size_t size, size_t count)
{
    if( size==8 )
    {
        while( count-- )
        {
            this->write( *((uint64_t *) ptr), 64 );
            ptr = ((uint64_t *) ptr) + 1;
        }
        return;
    }
    if( size==4 )
    {
        while( count-- )
        {
            this->write( *((uint32_t *) ptr), 32 );
            ptr = ((uint32_t *) ptr) + 1;
        }
        return;
    }
    if( size==2 )
    {
        while( count-- )
        {
            this->write( *((uint16_t *) ptr), 16 );
            ptr = ((uint16_t *) ptr) + 1;
        }
        return;
    }
    count *= size;
    while( count-- )
    {
        this->write( *((uint8_t *) ptr), 8 );
        ptr = ((uint8_t *) ptr) + 1;
    }
}

uint64_t membuf::grRead(uint_least8_t bits)
{
    uint64_t p = 0;

    while( this->read() )
    {
        p++;
        if( p>=32 )
            return this->read(32);
    }

    return (p<<bits) + this->read(bits);
}

void membuf::grWrite(uint64_t data, uint_least8_t bits)
{
    uint64_t p = data>>bits;

    if( p<32 )
    {
        this->write(MASK(p+1)-1, p+1);
        this->write(data&MASK(bits), bits);
    }
    else
    {
        this->write(MASK(32), 32);
        this->write(data, 32);
    }
}

std::chrono::nanoseconds membuf::rlgrRead(int64_t *seq, size_t N, uint_least8_t flagSigned) {
    auto start_time = std::chrono::high_resolution_clock::now();

    uint64_t u;
    uint64_t k_P = 0;
    uint64_t k_RP = 2 * L;
    uint64_t m = 0;
    uint64_t k;
    uint64_t k_R;
    uint64_t p;
    size_t n = 0;

    while (n < N) {
        k = k_P / L;
        k_R = k_RP / L;

        if (k) {
            // "Run" mode
            m = 0;
            
            while (this->read()) {
                m += 0x1 << k;
                k_P += U1;
                k = k_P / L;
            }
            
            m += this->read(k);
            
            while (m--)
                seq[n++] = 0;
            if (n >= N)
                break;

            u = this->grRead(k_R);

            seq[n++] = flagSigned ? _u2s(u + 1) : u + 1;
            p = u >> k_R;
            if (p) {
                k_RP += p - 1;
                if (k_RP > 32 * L)
                    k_RP = 32 * L;
            } else {
                if (k_RP < 2)
                    k_RP = 0;
                else
                    k_RP -= 2;
            }

            if (k_P < D1)
                k_P = 0;
            else
                k_P -= D1;
        } else {
            // "No run" mode
            u = this->grRead(k_R);
            
            seq[n++] = flagSigned ? _u2s(u) : u;
            p = u >> k_R;
            if (p) {
                k_RP = k_RP + p - 1;
                if (k_RP > 32 * L)
                    k_RP = 32 * L;
            } else {
                if (k_RP < 2)
                    k_RP = 0;
                else
                    k_RP -= 2;
            }
            if (u) {
                if (k_P < D0)
                    k_P = 0;
                else
                    k_P -= D0;
            } else
                k_P += U0;
        }
    }

    auto final_end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(final_end_time - start_time);
}

std::chrono::nanoseconds membuf::rlgrWrite(int64_t *seq, size_t N, uint_least8_t flagSigned) {
    auto start_time = std::chrono::high_resolution_clock::now();

    uint64_t u = 0;
    uint64_t k_P = 0;
    uint64_t k_RP = 2 * L;
    uint64_t m = 0;
    uint64_t k;
    uint64_t k_R;
    uint64_t p;

    for (size_t n = 0; n < N; n++) {
        u = flagSigned ? _s2u(seq[n]) : seq[n];
        k = k_P / L;
        k_R = k_RP / L;

        if (k) {
            // "Run" mode
            if (u) {
                u--;

                this->write(0);
                this->write(m, k);
                this->grWrite(u, k_R);

                p = u >> k_R;
                if (p) {
                    k_RP += p - 1;
                    if (k_RP > (32 * L))
                        k_RP = 32 * L;
                } else {
                    if (k_RP < 2)
                        k_RP = 0;
                    else
                        k_RP -= 2;
                }
                if (k_P < D1)
                    k_P = 0;
                else
                    k_P -= D1;
                m = 0;
            } else {
                m++;
                if (m == (0x1 << k)) {
                    this->write(1);
                    
                    k_P += U1;
                    m = 0;
                }
            }
        } else {
            // "No run" mode
            this->grWrite(u, k_R);

            p = u >> k_R;
            if (p) {
                k_RP = k_RP + p - 1;
                if (k_RP > 32 * L)
                    k_RP = 32 * L;
            } else {
                if (k_RP < 2)
                    k_RP = 0;
                else
                    k_RP -= 2;
            }
            if (u) {
                if (k_P < D0)
                    k_P = 0;
                else
                    k_P -= D0;
            } else
                k_P += U0;
            m = 0;
        }
    }

    if (k && !u) {
        this->write(0);
        this->write(m, k_P / L);
    }
    
    auto final_end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(final_end_time - start_time);
}
