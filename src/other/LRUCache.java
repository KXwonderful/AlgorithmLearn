package other;

import java.util.LinkedHashMap;

/**
 * 最近最好使用缓存机制算法 LRU
 */
public class LRUCache {

    // LRU 缓存算法的核心数据结构就是哈希链表 LinkedHashMap：双向链表和哈希表的结合体
    LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<>();
    int cap;

    // 若每次默认从链表尾部添加元素，那么越靠尾部的元素就是最近使用的，越靠头部的元素就是最久未使用的。
    // 对于某一个 key，可以通过哈希表快速定位到链表中的节点，从而取得对应 val。
    // 借助哈希表，可以通过 key 快速映射到任意一个链表节点，然后进行插入和删除

    public LRUCache(int capacity) {
        this.cap = capacity;
    }

    public int get(int key) {
        if (!cache.containsKey(key)) {
            return -1;
        }
        // 将 key 变为最近使用
        makeRecently(key);
        return cache.get(key);
    }

    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            // 修改 key 的值
            cache.put(key, value);
            // 将 key 变为最近使用
            makeRecently(key);
            return;
        }
        if (cache.size() >= this.cap) {
            // 链表头部就是最久未使用的 key
            int oldestKey = cache.keySet().iterator().next();
            cache.remove(oldestKey);
        }
        // 将新的 key 添加链表尾部
        cache.put(key, value);
    }

    private void makeRecently(int key) {
        int val = cache.get(key);
        // 删除 key，重新插入到队尾
        cache.remove(key);
        cache.put(key, val);
    }
}
