/**
 * @brief lightweight interface for using command line arguments
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <iomanip>
#include <ostream>
#include <variant>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include "iceicle/anomaly_log.hpp"

#ifdef ICEICLE_USE_PETSC
#include "petscsys.h"
#endif

namespace ICEICLE::UTIL::PROGRAM_ARGS {

    /// @brief a variant of all the types that can be parsed to
    using arg_variant = std::variant<std::monostate, std::string_view>;

    /**
     * @brief the stored value after parsing
     */
    struct parsed_value{

        std::optional<arg_variant> val;

        /** @brief check if this has a parsed value */
        inline constexpr bool has_value() const noexcept { return val.has_value(); }

        /** @brief check if this has a parsed value */
        inline constexpr operator bool() const noexcept { return has_value(); }

        /** @brief get the value given the value type 
         * This does not allow conversions of the tmplated type 
         * and thus is more strict than as()
         */
        template<class T>
        inline T &get(){ return std::get<T>(val.value()); }

        /** @brief get the value given the value type 
         * This does not allow conversions of the tmplated type 
         * and thus is more strict than as()
         */
        template<class T>
        inline const T &get() const { return std::get<T>(val.value());}

        /** @brief access the parsed value as the given type */
        template<class T>
        T as(){
            arg_variant& value = val.value();
            return std::visit([](auto &data){
                if constexpr( std::constructible_from<T, decltype(data)>){
                    return T{data};
                } else {
                    using namespace ICEICLE::UTIL;
                    std::string err_msg = std::string{"Cannot convert from "} + typeid(decltype(data)).name() 
                        + "to" + typeid(T).name();
                    AnomalyLog::log_anomaly(Anomaly{err_msg, general_anomaly_tag{} });
                    return T{};
                }
            }, value);
        }
    };

    // ================================
    // = Argument Type Specificiation =
    // ================================

    /// @brief tag specification of the type of the command line argument 
    template<class T>
    struct parse_type {
        using value_type = T;
    };

    /**
     * @brief A cli option 
     * in the command line inputs in the form --<name> are treated as options 
     * if this has a value type the next argument is parsed and stored as the specified value 
     * 
     * A special case is the void type which is a flag and doesn't parse the next agrument 
     * because there is no value to parse 
     *
     * @tparam optT the type of the option
     */
    template<class optT>
    struct cli_option{
        using value_type = optT;

        /// @brief the name of the option (comes before --)
        std::string name;

        /// @brief a description of the option 
        std::string_view desc;

        /// @brief the expected type
        parse_type<optT> expected_type{};

        /** @brief parse the value given the text */
        constexpr parsed_value parse(std::string_view text) const noexcept;

    };

    /// @brief special case of an option with no value
    using cli_flag = cli_option<void>;

    // === parse specializations ===
    template<>
    inline constexpr parsed_value cli_option<void>::parse(
        std::string_view text) const noexcept
    { return parsed_value{std::monostate{}}; }

    template<>
    inline constexpr parsed_value cli_option<std::string_view>::parse(
        std::string_view text) const noexcept
    { return parsed_value{text}; }

    /**
     * @brief parser for command line arguments 
     */
    class cli_parser {
        std::size_t max_arg_size = 0;

        /** @brief map of the options by name to their parsed values */
        std::map<std::string, parsed_value> options;

        /** @brief map of option form (--<name>) strings in argv to their respective indices */
        std::map<std::string, int> in_options;

        /** @brief the descriptions of each option */
        std::map<std::string, std::string> descriptions;

        int argc;
        char **argv;

        template<class optT>
        void add_option(const cli_option<optT> &opt){
            max_arg_size = std::max(max_arg_size, opt.name.size());
            // if the option has been specified by the user:
            // parse it 
            //
            // otherwise: put it in the map but give no value 
            // so has_value() or bool() evaluate to false
            if(in_options.contains(opt.name)){
                // if this is a flag then there is no text to parse
                std::string_view to_parse = (std::same_as<optT, void>) ? "" : argv[in_options[opt.name] + 1];
                options[opt.name] = opt.parse(to_parse);
            } else {
                // value remains unfilled (evaluates to fase)
                options[opt.name] = parsed_value{};
            }
            descriptions[opt.name] = opt.desc;
        }

        public:

        /**
         * @brief construct a cli parser 
         * @param argc the argument count 
         * @param argv the argument values 
         *
         * NOTE: this expects command line arguments so argc[0] is generally ignored
         * as that contains the program name
         */
        cli_parser(int argc, char* argv[]) : argc(argc), argv(argv) {
#ifdef ICEICLE_USE_PETSC
            // suppress petsc unused argument warnings
            // we have our own argument parser 
            PetscOptionsSetValue(NULL, "-options_left", "false");
#endif
            using namespace ICEICLE::UTIL;
            // fill the in_options map
            for(int i = 1; i < argc; ++i){
                std::string_view arg_view = argv[i];
                if(arg_view.starts_with("--")){
                    std::string key{arg_view.substr(2)};
                    if(in_options.contains(key)){
                        std::string err_str = "option: " + key + " is already specified";
                        AnomalyLog::log_anomaly(Anomaly{err_str, warning_anomaly_tag{}});
                    } else {
                        in_options[key] = i;
                    }
                }
            }
        }

        /**
         * @brief add command line options 
         * @param options the options to add 
         * use cli_option{} to create a new option with a value 
         * or cli_flag{} to create a new flag (which does not have an associated value)
         * cli_flag is an alias for cli_option<void>
         */
        template<class... optTs>
        inline void add_options(const cli_option<optTs>&... options){
            (add_option(options), ...); // sexiest fold expression in the west
        }

        /**
         * @brief equivalent to std::map::operator[] 
         * accesses the value indexed by key or inserts if the key does not exist
         * new parsed_value instances cast to boolean false and has_value() evaluates to false 
         * so this can still be used to check for the value
         */
        inline parsed_value& operator[](const std::string& key){
            return options[key];
        }

        /** @brief print the options with their provided descriptions */
        inline void print_options(std::ostream &os) const {
            os << "Available options (use --<option_name> to use):" << std::endl;
            for(auto &kvpair : descriptions){
                os << std::setw(max_arg_size) << std::left << kvpair.first << ": " << kvpair.second << std::endl;
            }
            os << std::endl;
        }

    };

}
