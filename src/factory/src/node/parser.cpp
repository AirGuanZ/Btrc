#include <ranges>

#include <nlohmann/json.hpp>

#include <btrc/factory/node/parser.h>
#include <btrc/utils/file.h>

BTRC_FACTORY_BEGIN

namespace
{

    namespace js = nlohmann;

    using std::filesystem::path;

    RC<Node> json_to_node(const js::ordered_json &json)
    {
        if(json.is_object())
        {
            auto ret = newRC<Group>();
            for(auto it = json.begin(); it != json.end(); ++it)
            {
                const auto &val = it.value();
                ret->insert(it.key(), json_to_node(val));
            }
            return ret;
        }
        if(json.is_array())
        {
            auto ret = newRC<Array>();
            for(auto &elem : json)
                ret->push_back(json_to_node(elem));
            return ret;
        }
        auto ret = newRC<Value>();
        if(json.is_boolean())
            ret->set_string(json.get<bool>() ? "true" : "false");
        else if(json.is_number_float())
            ret->set_string(std::to_string(json.get<float>()));
        else if(json.is_number_integer())
            ret->set_string(std::to_string(json.get<int>()));
        else
            ret->set_string(json.get<std::string>());
        return ret;
    }

} // namespace anonymous

void JSONParser::set_source(std::string src)
{
    src_ = std::move(src);
}

void JSONParser::add_include_directory(const path &dir)
{
    include_dirs_.insert(absolute(dir).lexically_normal());
}

void JSONParser::parse()
{
    process_includes();

    const auto json = js::ordered_json::parse(src_, nullptr, true, true);
    result_ = json_to_node(json)->as_group();
    if(!result_)
        throw BtrcException("root node is not a group");

    std::vector<RC<Node>> node_path = { result_ };
    for(auto &[key, value] : *result_)
        resolve_references(node_path, value);
}

RC<Group> JSONParser::get_result()
{
    return result_;
}

path JSONParser::get_absolute_included_file_path(const path &included_file) const
{
    path final_included_file;

    if(included_file.is_relative())
    {
        for(auto &dir : include_dirs_)
        {
            auto abs_dir = dir / included_file;
            if(exists(abs_dir))
            {
                final_included_file = std::move(abs_dir);
                break;
            }
        }
        if(final_included_file.empty())
        {
            throw BtrcException(std::format(
                "included file '{}' is not found", included_file.string()));
        }
        final_included_file = final_included_file.lexically_normal();
    }
    else
        final_included_file = included_file.lexically_normal();

    return final_included_file;
}

void JSONParser::process_includes()
{
    for(;;)
    {
        size_t beg = src_.find("\"$include{");
        if(beg == std::string::npos)
            break;

        beg += 10;
        const size_t end = src_.find("}\"", beg);
        if(end == std::string::npos)
        {
            throw BtrcException(
                "failed to find the enclosing '}' of a '$include{'");
        }

        const auto included_file = src_.substr(beg, end - beg);
        const auto final_included_file = get_absolute_included_file_path(included_file);
        const auto included = read_txt_file(final_included_file.string());

        const auto prefix = src_.substr(0, beg - 10);
        const auto suffix = src_.substr(end + 2);

        src_ = prefix;
        src_ += included;
        src_ += suffix;
    }
}

void JSONParser::resolve_references(std::vector<RC<Node>> &current_path, RC<Node> &node)
{
    if(auto group = node->as_group())
    {
        current_path.push_back(node);
        for(auto &[key, value] : *group)
            resolve_references(current_path, value);
        current_path.pop_back();
    }
    else if(auto arr = node->as_array())
    {
        current_path.push_back(node);
        for(auto &elem : *arr)
            resolve_references(current_path, elem);
        current_path.pop_back();
    }
    else
    {
        auto val = node->as_value();
        assert(val);
        auto &str = val->get_string();
        if(str.find("$reference{") == 0)
        {
            if(str[str.length() - 1] != '}')
                throw BtrcException("'}' for '$reference{' is not found");
            const auto path = str.substr(11, str.length() - 12);
            node = find_node(current_path, path);
        }
    }
}

RC<Node> JSONParser::find_node(std::vector<RC<Node>> &current_path, const std::string &path)
{
    auto ss = path
        | std::ranges::views::split('/')
        | std::ranges::views::transform(
            [](auto &&s) { return std::string_view(s); });

    std::vector<std::string_view> secs;
    for(auto s : ss)
        secs.push_back(s);
    if(secs.empty())
        throw BtrcException("invalid node path: " + path);

    auto &start_sec = secs.front();
    int start_index;
    if(start_sec == "$root")
        start_index = 0;
    else
    {
        start_index = static_cast<int>(current_path.size() - 1);
        while(start_index >= 0)
        {
            RC<Node> node = current_path[start_index];
            if(auto group = node->as_group())
            {
                if(group->find_child_node(start_sec))
                    break;
            }
            else if(auto arr = node->as_array())
            {
                try
                {
                    const size_t index = std::stoul(std::string(start_sec));
                    if(index < arr->get_size())
                        break;
                }
                catch(...)
                {

                }
            }
            --start_index;
        }
    }
    if(start_index < 0)
        throw BtrcException("invalid node path: " + path);

    std::vector<RC<Node>> new_path;
    for(int j = 0; j < start_index; ++j)
        new_path.push_back(current_path[j]);

    auto node = current_path[start_index];
    for(size_t i = 0; i < secs.size(); ++i)
    {
        auto sec = secs[i];
        if(i == 0 && sec == "$root")
            continue;
        new_path.push_back(node);
        if(auto grp = node->as_group())
        {
            node = grp->find_child_node(sec);
            if(!node)
                throw BtrcException("invalid node path: " + path);
        }
        else if(auto arr = node->as_array())
        {
            size_t index = 0;
            try
            {
                index = std::stoul(std::string(sec));
            }
            catch(...)
            {
                throw BtrcException("invalid node path: " + path);
            }
            if(index >= arr->get_size())
                throw BtrcException("invalid node path: " + path);
            node = arr->get_element(index);
        }
        else
            throw BtrcException("invalid node path: " + path);
    }

    if(auto val = node->as_value();
       val && val->get_string().find("$reference{") != std::string::npos)
    {
        auto &str = val->get_string();
        if(str[str.length() - 1] != '}')
            throw BtrcException("'}' for '$reference{' is not found");
        node = find_node(current_path, str.substr(11, str.length() - 12));
    }

    return node;
}

BTRC_FACTORY_END
