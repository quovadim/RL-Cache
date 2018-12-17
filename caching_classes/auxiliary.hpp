#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <map>
#include <vector>
#include <random>
#include <algorithm>

const double WINDOW = 0.99999;

using std::map;
using std::vector;
using std::exp;
using std::pair;
using std::multimap;

namespace p = boost::python;

inline
bool value_comparer(const pair<uint64_t, double> &i1, const pair<uint64_t, double> &i2) {
	return i1.first <= i2.first;
}

template< typename T >
inline
std::vector< T > to_std_vector( const p::object& iterable )
{
    return std::vector< T >( p::stl_input_iterator< T >( iterable ),
                             p::stl_input_iterator< T >( ) );
}

template<class T>
inline
p::list to_py_list(const std::vector<T>& v)
{
    p::object get_iter = p::iterator<std::vector<T> >();
    p::object iter = get_iter(v);
    p::list l(iter);
    return l;
}

template <class K, class V>
inline
boost::python::dict to_py_dict(std::map<K, V> map) {
    typename std::map<K, V>::iterator iter;
	p::dict dictionary;
	for (iter = map.begin(); iter != map.end(); ++iter) {
		dictionary[iter->first] = iter->second;
	}
	return dictionary;
}

template <class K, class V>
inline
std::map<K, V> to_std_map(p::dict dict) {
    std::map<K, V> result;
    p::list keys = dict.keys();
    p::list values = dict.values();
    uint64_t keys_len = p::len(keys);
	for (uint64_t i = 0; i < keys_len; i++) {
	    K key = p::extract<K>(keys[i]);
	    V value = p::extract<V>(values[i]);
		result.insert(pair<K, V>(key, value));
	}
	return result;
}

template <class K, class V>
inline
std::multimap<K, V> to_std_multimap(p::dict dict) {
    std::multimap<K, V> result;
    p::list keys = dict.keys();
    p::list values = dict.values();
    uint64_t keys_len = p::len(keys);
	for (uint64_t i = 0; i < keys_len; i++) {
	    K key = p::extract<K>(keys[i]);
	    V value = p::extract<V>(values[i]);
		result.insert(pair<K, V>(key, value));
	}
	return result;
}

template <class K, class V>
inline
boost::python::dict mm_to_py_dict(std::multimap<K, V> map) {
    typename std::multimap<K, V>::iterator iter;
	p::dict dictionary;
	for (iter = map.begin(); iter != map.end(); ++iter) {
		dictionary[iter->first] = iter->second;
	}
	return dictionary;
}

#define PYTHON_ERROR(TYPE, REASON) \
{ \
    PyErr_SetString(TYPE, REASON); \
    throw p::error_already_set(); \
}

template<class T>
inline PyObject * managingPyObject(T *_p)
{
    return typename p::manage_new_object::apply<T *>::type()(_p);
}

template<class Copyable>
p::object
generic__copy__(p::object copyable)
{
    Copyable *newCopyable(new Copyable(p::extract<const Copyable
&>(copyable)));
    p::object
result(p::detail::new_reference(managingPyObject(newCopyable)));

    p::extract<p::dict>(result.attr("__dict__"))().update(
        copyable.attr("__dict__"));

    return result;
}

template<class Copyable>
p::object
generic__deepcopy__(p::object copyable, p::dict memo)
{
    p::object copyMod = p::import("copy");
    p::object deepcopy = copyMod.attr("deepcopy");

    Copyable *newCopyable(new Copyable(p::extract<const Copyable
&>(copyable)));
    p::object
result(p::detail::new_reference(managingPyObject(newCopyable)));

    // HACK: copyableId shall be the same as the result of id(copyable) in Python -
    // please tell me that there is a better way! (and which ;-p)
    uint64_t copyableId = reinterpret_cast<uint64_t>(copyable.ptr());
    memo[copyableId] = result;

    p::extract<p::dict>(result.attr("__dict__"))().update(
        deepcopy(p::extract<p::dict>(copyable.attr("__dict__"))(),
memo));

    return result;
}

struct CacheObject {
    uint64_t id;
    uint64_t size;
    uint64_t timestamp;

    CacheObject(p::dict &request) {
        id = p::extract<uint64_t>(request.get("id"));
        size = p::extract<uint64_t>(request.get("size"));
        timestamp = p::extract<uint64_t>(request.get("timestamp"));
    }
};