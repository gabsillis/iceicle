#include <algorithm>
#include <stdexcept>
#include <string>
#include <limits.h>
#include <string_view>
namespace iceicle {

    /// @brief similar interface to std::bitset 
    ///
    /// represents a fixed size sequence of bits 
    /// @tparam nbit the number of bits in the sequence 
    /// effectively limited at the number of bits in unsigned long long
    ///
    /// positions in the sequence (counting from 0) are from least significant (right) 
    /// to most significant (left) of the set
    ///
    /// actually constexpr because for some reason 
    /// no one wants to make standards compliant implementations
    /// :3
    ///
    /// This implementation is not equivalent to std::bitset and makes some simplifications
    template<std::size_t nbit>
    class bitset {
        public:

        using data_t = unsigned long long;
        static constexpr std::size_t _NBITS_IN_SET = CHAR_BIT * sizeof(data_t);

        static constexpr auto _which_bit(std::size_t pos)
            { return pos % _NBITS_IN_SET; }

        /// @brief mask a bit at the given index 
        /// @param pos the index
        static constexpr 
        auto _mask_bit(std::size_t pos) -> data_t
        { return static_cast<data_t>(1) << _which_bit(pos); }

        /// @brief check if position is within bitset 
        /// @throws std::out_of_range if out of range
        static constexpr 
        auto _check_pos(std::size_t pos) -> void 
        { if(pos >= nbit) throw std::out_of_range("Position is not within bitset"); }

        /// @brief mask for all of the bits in the sequence
        static constexpr data_t full_set_mask = ~static_cast<data_t>(0) >> (64 - nbit) ; // NOTE: this works for 1-64 

        public:

        /// @brief reference to a specific bit
        class reference {
            friend class bitset;
            private:
            data_t *ptr;
            std::size_t bpos;

            /// @brief constructor takes pointer to the data and bit position index
            constexpr 
            reference(data_t *ptr, std::size_t bpos) noexcept 
            : ptr{ptr}, bpos{bpos} {}

            public:

            /// @brief set the value that is referenced to x
            /// b[i] = x
            constexpr
            auto operator=(bool x) noexcept -> reference&
            {
                if(x)
                    *ptr |= _mask_bit(bpos);
                else 
                    *ptr &= ~_mask_bit(bpos);
                return *this;
            }

            /// @brief assignment to the value of another bit
            /// b[i] = b[j]
            constexpr 
            auto operator=(const reference& other) noexcept {
                if( (bool) other )
                    *ptr |= _mask_bit(bpos);
                else 
                    *ptr &= ~_mask_bit(bpos);
            };

            /// @brief convert the referenced bit to a bool
            constexpr 
            operator bool() const noexcept 
            { return ((*ptr) & _mask_bit(bpos)) != 0; }

            /// @brief get the inverse of the referenced bit as a bool
            constexpr
            auto operator~() const noexcept -> bool
            { return ((*ptr) & _mask_bit(bpos)) == 0; }

            /// @brief flip the value of the bit referenced 
            constexpr 
            auto flip() noexcept -> reference&
            { 
                *ptr ^= _mask_bit(bpos);
                return *this;
            }

            /// @brief destructor
            constexpr
            ~reference() = default;
        };

        /// @brief the data to represent the stored bits
        data_t _data;

        /// @brief default constructor initializes to zero
        constexpr 
        bitset() noexcept : _data{0} {}

        /// @brief construct from an unsigned long long truncated to value_t size
        constexpr 
        bitset(unsigned long long val) noexcept 
        : _data{val & full_set_mask} {}

        /// @brief construct from a string view
        constexpr explicit 
        bitset(std::string_view str) noexcept
        : _data{0} {
            for(int i = 0; i < std::min(nbit, str.length()); ++i)
            { if(str[str.length() - i - 1] == '1') unchecked_set_true(i); }
        }

        /// @brief use strong ordering
        /// spaceship! ^.^
        constexpr 
        auto operator<=>(const bitset&) const = default;

        // ===========================
        // = Access and Modification =
        // ===========================

        /// @brief get the value at the given position
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto operator[]( std::size_t pos ) const noexcept -> bool
        { return (_data & _mask_bit(pos)) != 0; }

        /// @brief access a modifiable reference to the bit as position pos
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto operator[]( std::size_t pos) noexcept -> reference 
        { return reference(&_data, pos); }

        /// @brief returns the value of the bit at posiition pos
        constexpr 
        auto test( std::size_t pos ) const -> bool {
            _check_pos(pos);
            return (_data & _mask_bit(pos)) != 0;
        }

        /// @brief check if all the bits in the set are set to true
        constexpr 
        auto all() const noexcept -> bool 
        { return (_data & full_set_mask) == full_set_mask; }

        /// @brief check if any of the bits in the set are true
        constexpr 
        auto any() const noexcept -> bool 
        { return (_data & full_set_mask) != 0; }

        /// @brief check if none of the bits in the set are true
        constexpr 
        auto none() const noexcept -> bool 
        { return (_data & full_set_mask) == 0; }

        /// @brief count the number of set bits 
        /// https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSet64
        /// TODO: maybe find a way to optimize for specifically nbit bits 
        /// and eliminate mask
        constexpr 
        auto count() const noexcept -> std::size_t 
        { 
            data_t v = _data & full_set_mask;
            v = v - ((v >> 1) & (data_t)~(data_t)0/3);                           // temp
            v = (v & (data_t)~(data_t)0/15*3) + ((v >> 2) & (data_t)~(data_t)0/15*3);      // temp
            v = (v + (v >> 4)) & (data_t)~(data_t)0/255*15;                      // temp
            return (data_t)(v * ((data_t)~(data_t)0/255)) >> (sizeof(data_t) - 1) * CHAR_BIT;
        }

        /// @brief get the number of bits the bitset represents
        constexpr 
        auto size() const noexcept -> std::size_t 
        { return nbit; }

        /// @brief set the bits to the result of binary AND of *this and other
        constexpr 
        auto operator&=( const bitset<nbit>& other ) noexcept -> bitset&
        {
            _data &= other._data;
            return *this;
        }

        /// @brief set the bits to the result of binary OR of *this and other
        constexpr 
        auto operator|=( const bitset<nbit>& other ) noexcept -> bitset&
        {
            _data |= other._data;
            return *this;
        }

        /// @brief set the bits to the result of binary XOR of *this and other
        constexpr 
        auto operator^=( const bitset<nbit>& other ) noexcept -> bitset&
        {
            _data ^= other._data;
            return *this;
        }

        /// @brief return a temporary copy of this with bits flipped (binary NOT)
        constexpr 
        auto operator~() const noexcept -> bitset 
        { return bitset<nbit>{_data}.flip(); }

        /// @brief return a temporary copy binary left shifted by pos bits
        constexpr 
        auto operator<<( std::size_t pos ) const noexcept -> bitset 
        { return bitset<nbit>{_data << pos}; }

        /// @brief perform a left shift on the data in *this
        constexpr
        auto operator<<=( std::size_t pos ) noexcept -> bitset&
        {
            _data <<= pos;
            return *this;
        }

        /// @brief return a temporary copy binary right shifted by pos bits
        constexpr 
        auto operator>>( std::size_t pos ) const noexcept -> bitset 
        { return bitset<nbit>{_data >> pos}; }

        /// @brief perform a left shift on the data in *this
        constexpr
        auto operator>>=( std::size_t pos ) noexcept -> bitset&
        {
            _data >>= pos;
            return *this;
        }

        /// @brief set all bits to true
        constexpr 
        auto set() noexcept -> bitset& 
        { 
            _data = ~static_cast<data_t>(0); 
            return *this;
        }

        /// @brief set the bit at position pos to true
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto unchecked_set_true( std::size_t pos ) -> bitset&
        {
            _data |= _mask_bit(pos); 
            return *this;
        }

        /// @brief set the bit at position pos to the given value
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto unchecked_set( std::size_t pos, bool value = true ) noexcept -> bitset&
        {
            if(value)
                _data |= _mask_bit(pos); 
            else
                _data &= ~_mask_bit(pos);
            return *this;
        }

        /// @brief set the bit at position pos to the given value
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        ///
        /// @throws std::out_of_range if pos does not correspond to a valid position in the bitset
        constexpr 
        auto set( std::size_t pos, bool value = true ) -> bitset&
        {
            _check_pos(pos);
            return unchecked_set(pos, value);
        }

        /// @brief set all the bits to false
        constexpr
        auto reset() noexcept -> bitset&
        { 
            _data = static_cast<data_t>(0); 
            return *this;
        }

        /// @brief set the bit at pos to false
        constexpr 
        auto reset( std::size_t pos ) noexcept -> bitset& 
        {
            _data &= ~_mask_bit(pos);
            return *this;
        }

        /// @brief flip the bits (in place)
        constexpr 
        auto flip() noexcept -> bitset& 
        {
            _data = ~_data;
            return *this;
        }

        /// @brief flip the bit at the position pos 
        /// @throws sstd::out_of_range if pos does not correspond to a valid position within the bitset
        constexpr 
        auto flip( std::size_t pos ) -> bitset&
        {
            _check_pos(pos);
            _data ^= _mask_bit(pos);
            return *this;
        }

        // ===============
        // = Conversions =
        // ===============

        /// @brief get a string representation of the bitset
        template<
            class CharT = char,
            class Traits = std::char_traits<CharT>,
            class Allocator = std::allocator<CharT>
        >
        constexpr 
        auto to_string( char zero = '0', char one = '1') const noexcept -> std::basic_string<CharT, Traits, Allocator>
        {
            std::basic_string<CharT, Traits, Allocator> result;
            result.assign(nbit, zero);
            for(std::size_t pos = nbit - 1; pos != -1; --pos) {
                if( (_mask_bit(pos) & _data) != static_cast<data_t>(0) ){
                    result[nbit - pos - 1] = one;
                }
            }
            return result;
        }

        /// @brief get the contents of the bitset as an unsigned long
        constexpr 
        auto to_ulong() const -> unsigned long 
        { return static_cast<unsigned long>(_data); }


        /// @brief get the contents of the bitset as an unsigned long long
        constexpr 
        auto to_ullong() const -> unsigned long long
        { return static_cast<unsigned long long>(_data); }
    };

    /// @brief specialization for zero size bitset 
    /// The data field is always 0 and static so this has zero memory overhead
    template<>
    class bitset<0> {
        public:

        using data_t = unsigned long long;
        static constexpr std::size_t _NBITS_IN_SET = CHAR_BIT * sizeof(data_t);

        static constexpr auto _which_bit(std::size_t pos)
            { return pos % _NBITS_IN_SET; }

        /// @brief mask a bit at the given index 
        /// @param pos the index
        static constexpr 
        auto _mask_bit(std::size_t pos) -> data_t
        { return static_cast<data_t>(1) << _which_bit(pos); }

        /// @brief check if position is within bitset 
        /// @throws std::out_of_range if out of range
        static
        auto _check_pos(std::size_t pos) -> void 
        { throw std::out_of_range("Position is not within bitset"); }

        /// @brief mask for all of the bits in the sequence
        static constexpr data_t full_set_mask = static_cast<data_t>(0) ;

        public:

        /// @brief reference to a specific bit
        class reference {
            friend class bitset;
            private:
            const data_t *ptr;
            std::size_t bpos;

            /// @brief constructor takes pointer to the data and bit position index
            constexpr 
            reference(const data_t *ptr, std::size_t bpos) noexcept 
            : ptr{ptr}, bpos{bpos} {}

            public:

            /// @brief set the value that is referenced to x
            /// b[i] = x
            constexpr
            auto operator=(bool x) noexcept -> reference&
            { return *this; }

            /// @brief assignment to the value of another bit
            /// b[i] = b[j]
            constexpr 
            auto operator=(const reference& other) noexcept {};

            /// @brief convert the referenced bit to a bool
            constexpr 
            operator bool() const noexcept 
            { return ((*ptr) & _mask_bit(bpos)) != 0; }

            /// @brief get the inverse of the referenced bit as a bool
            constexpr
            auto operator~() const noexcept -> bool
            { return ((*ptr) & _mask_bit(bpos)) == 0; }

            /// @brief flip the value of the bit referenced 
            constexpr 
            auto flip() noexcept -> reference&
            { return *this; }

            /// @brief destructor
            constexpr
            ~reference() = default;
        };

        /// @brief the data to represent the stored bits
        static constexpr data_t _data = 0;

        /// @brief default constructor initializes to zero
        constexpr 
        bitset() noexcept {}

        /// @brief construct from an unsigned long long truncated to value_t size
        constexpr 
        bitset(unsigned long long val) noexcept {}

        /// @brief construct from a string view
        constexpr explicit 
        bitset(std::string_view str) noexcept {}

        /// @brief use strong ordering
        /// spaceship! ^.^
        constexpr 
        auto operator<=>(const bitset&) const = default;

        // ===========================
        // = Access and Modification =
        // ===========================

        /// @brief get the value at the given position
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto operator[]( std::size_t pos ) const noexcept -> bool
        { return _data; }

        /// @brief access a modifiable reference to the bit as position pos
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto operator[]( std::size_t pos) noexcept -> reference 
        { return reference(&_data, pos); }

        /// @brief returns the value of the bit at posiition pos
        constexpr 
        auto test( std::size_t pos ) const -> bool {
            _check_pos(pos);
            return (_data & _mask_bit(pos)) != 0;
        }

        /// @brief check if all the bits in the set are set to true
        constexpr 
        auto all() const noexcept -> bool 
        { return (_data & full_set_mask) == full_set_mask; }

        /// @brief check if any of the bits in the set are true
        constexpr 
        auto any() const noexcept -> bool 
        { return (_data & full_set_mask) != 0; }

        /// @brief check if none of the bits in the set are true
        constexpr 
        auto none() const noexcept -> bool 
        { return (_data & full_set_mask) == 0; }

        /// @brief count the number of set bits 
        /// https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSet64
        constexpr 
        auto count() const noexcept -> std::size_t 
        { 
            data_t v = _data;
            v = v - ((v >> 1) & (data_t)~(data_t)0/3);                           // temp
            v = (v & (data_t)~(data_t)0/15*3) + ((v >> 2) & (data_t)~(data_t)0/15*3);      // temp
            v = (v + (v >> 4)) & (data_t)~(data_t)0/255*15;                      // temp
            return (data_t)(v * ((data_t)~(data_t)0/255)) >> (sizeof(data_t) - 1) * CHAR_BIT;
        }

        /// @brief get the number of bits the bitset represents
        constexpr 
        auto size() const noexcept -> std::size_t 
        { return 0; }

        /// @brief set the bits to the result of binary AND of *this and other
        constexpr 
        auto operator&=( const bitset<0>& other ) noexcept -> bitset&
        { return *this; }

        /// @brief set the bits to the result of binary OR of *this and other
        constexpr 
        auto operator|=( const bitset<0>& other ) noexcept -> bitset&
        { return *this; }

        /// @brief set the bits to the result of binary XOR of *this and other
        constexpr 
        auto operator^=( const bitset<0>& other ) noexcept -> bitset&
        { return *this; }

        /// @brief return a temporary copy of this with bits flipped (binary NOT)
        constexpr 
        auto operator~() const noexcept -> bitset 
        { return bitset<0>{}; }

        /// @brief return a temporary copy binary left shifted by pos bits
        constexpr 
        auto operator<<( std::size_t pos ) const noexcept -> bitset 
        { return bitset<0>{}; }

        /// @brief perform a left shift on the data in *this
        constexpr
        auto operator<<=( std::size_t pos ) noexcept -> bitset&
        { return *this; }

        /// @brief return a temporary copy binary right shifted by pos bits
        constexpr 
        auto operator>>( std::size_t pos ) const noexcept -> bitset 
        { return bitset<0>{}; }

        /// @brief perform a left shift on the data in *this
        constexpr
        auto operator>>=( std::size_t pos ) noexcept -> bitset&
        { return *this; }

        /// @brief set all bits to true
        constexpr 
        auto set() noexcept -> bitset& 
        { return *this; }

        /// @brief set the bit at position pos to true
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto unchecked_set( std::size_t pos ) -> bitset&
        { return *this; }

        /// @brief set the bit at position pos to the given value
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        constexpr 
        auto unchecked_set( std::size_t pos, bool value = true ) noexcept -> bitset&
        { return *this; }

        /// @brief set the bit at position pos to the given value
        /// @param pos the position (counting from 0) from least significant (right) 
        /// to most significant (left) of the set
        ///
        /// @throws std::out_of_range if pos does not correspond to a valid position in the bitset
        constexpr 
        auto set( std::size_t pos, bool value = true ) -> bitset&
        {
            _check_pos(pos);
            return unchecked_set(pos, value);
        }

        /// @brief set all the bits to false
        constexpr
        auto reset() noexcept -> bitset&
        { return *this; }

        /// @brief set the bit at pos to false
        constexpr 
        auto reset( std::size_t pos ) noexcept -> bitset& 
        { return *this; }

        /// @brief flip the bits (in place)
        constexpr 
        auto flip() noexcept -> bitset& 
        { return *this; }

        /// @brief flip the bit at the position pos 
        /// @throws sstd::out_of_range if pos does not correspond to a valid position within the bitset
        constexpr 
        auto flip( std::size_t pos ) -> bitset&
        {
            _check_pos(pos);
            return *this;
        }

        // ===============
        // = Conversions =
        // ===============

        /// @brief get a string representation of the bitset
        template<
            class CharT = char,
            class Traits = std::char_traits<CharT>,
            class Allocator = std::allocator<CharT>
        >
        constexpr 
        auto to_string( char zero = '0', char one = '1') const noexcept -> std::basic_string<CharT, Traits, Allocator>
        {
            std::basic_string<CharT, Traits, Allocator> result;
            result.assign(0, zero);
            return result;
        }

        /// @brief get the contents of the bitset as an unsigned long
        constexpr 
        auto to_ulong() const -> unsigned long 
        { return static_cast<unsigned long>(_data); }


        /// @brief get the contents of the bitset as an unsigned long long
        constexpr 
        auto to_ullong() const -> unsigned long long
        { return static_cast<unsigned long long>(_data); }
    };

    template<std::size_t N>
    constexpr 
    auto operator&( const bitset<N>& lhs, const bitset<N>& rhs) -> bitset<N>
    { return bitset<N>{lhs._data & rhs._data}; }

    template<std::size_t N>
    constexpr 
    auto operator|( const bitset<N>& lhs, const bitset<N>& rhs) -> bitset<N>
    { return bitset<N>{lhs._data | rhs._data}; }

    template<std::size_t N>
    constexpr 
    auto operator^( const bitset<N>& lhs, const bitset<N>& rhs) -> bitset<N>
    { return bitset<N>{lhs._data ^ rhs._data}; }
}
