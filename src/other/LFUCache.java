package other;

import java.util.HashMap;
import java.util.LinkedHashSet;

/**
 * 最不经常使用缓存机制算法 LFU
 */
public class LFUCache {

    // 核心思想是淘汰访问频次最低的数据，如果访问频次最低的数据有多条，需要淘汰最旧的数据。

    // 使用一个 HashMap 存储 key 到 val 的映射，就可以快速计算 get(key)
    HashMap<Integer, Integer> keyToVal;
    // 使用一个 HashMap 存储 key 到 freq 的映射，就可以快速操作 key 对应的 freq
    HashMap<Integer, Integer> keyToFreq;
    // 使用一个 HashMap 存储 freq 到 key列表 的映射，其中 LinkedHashSet 是链表和哈希集合的结合体
    HashMap<Integer, LinkedHashSet<Integer>> freqToKeys;
    // 记录最小的频次
    int minFreq;
    // 记录 LFU 缓存的最大容量
    int cap;

    /**
     * freq 到 key 列表的映射 -- FK表
     * <p>
     * 1、首先，肯定是需要freq到key的映射，用来找到freq最小的key。
     * 2、将freq最小的key删除，就得快速得到当前所有key最小的freq是多少。
     * 想要时间复杂度 O(1) 的话，需要用一个变量minFreq来记录当前最小的freq。
     * 3、可能有多个key拥有相同的freq，所以 freq对key是一对多的关系，即一个freq对应一个key的列表。
     * 4、希望freq对应的key的列表是存在时序的，便于快速查找并删除最旧的key。
     * 5、希望能够快速删除key列表中的任何一个key，因为如果频次为freq的某个key被访问，
     * 那么它的频次就会变成freq+1，就应该从freq对应的key列表中删除，加到freq+1对应的key的列表中。
     */


    // 构造容量为 capacity 的缓存
    public LFUCache(int capacity) {
        keyToVal = new HashMap<>();
        keyToFreq = new HashMap<>();
        freqToKeys = new HashMap<>();
        this.cap = capacity;
        this.minFreq = 0;
    }

    // 在缓存中查询 key
    public int get(int key) {
        if (!keyToVal.containsKey(key)) {
            return -1;
        }
        // 增加 key 对应的 freq
        increaseFreq(key);
        return keyToVal.get(key);
    }

    // 将 key 和 val 存入缓存
    public void put(int key, int value) {
        if (cap <= 0) return;

        // 若 key 已存在，修改对应的 val 即可
        if (keyToVal.containsKey(key)) {
            keyToVal.put(key, value);
            increaseFreq(key);
            return;
        }

        // key 不存在，需要插入
        // 容量已满的话需要淘汰一个 freq 最小的 key
        if (keyToVal.size() >= cap) {
            removeMinFreqKey();
        }

        // 插入新的 key 和 val，对应的 freq 为 1
        keyToVal.put(key, value);
        keyToFreq.put(key, 1);
        freqToKeys.putIfAbsent(1, new LinkedHashSet<>());
        freqToKeys.get(1).add(key);
        // 插入新 key 后最小的 freq 肯定是 1
        minFreq = 1;
    }

    /**
     * 删除最小频率的 key
     * <p>
     * 删除某个键 key要同时修改三个映射表，借助 minFreq 参数可以从 FK表 中找到 freq 最小的 keyList，
     * 根据时序，其中第一个元素就是要被淘汰的 deletedKey，操作三个映射表删除这个 key 即可
     */
    private void removeMinFreqKey() {
        // freq 最小的 key 列表
        LinkedHashSet<Integer> keyList = freqToKeys.get(minFreq);
        // 其中最先被插入的那个 key 就是该被淘汰的 key
        int deleteKey = keyList.iterator().next();
        // 更新 FK 表
        keyList.remove(deleteKey);
        if (keyList.isEmpty()) {
            freqToKeys.remove(minFreq);
        }
        // 更新 KV 表
        keyToVal.remove(deleteKey);
        // 更新 KF 表
        keyToFreq.remove(deleteKey);
    }

    /**
     * 增加 key 对应的 freq
     * <p>
     * 更新某个 key 的 freq 会涉及 FK表 和 KF表，这边分别更新这两个表即可
     */
    private void increaseFreq(int key) {
        int freq = keyToFreq.get(key);
        keyToFreq.put(key, freq + 1);

        // 更新 FK 表
        // 将 key 从 freq 对应的列表中删除
        freqToKeys.get(freq).remove(key);
        // 将 key 加入 freq + 1 对应的列表中
        freqToKeys.putIfAbsent(freq + 1, new LinkedHashSet<>());
        freqToKeys.get(freq + 1).add(key);

        // 如果 freq 对应的列表空了，移除这个 freq
        if (freqToKeys.get(freq).isEmpty()) {
            freqToKeys.remove(freq);
            // 如果这个 freq 恰好是 minFreq，更新 minFreq
            if (freq == minFreq) {
                minFreq++;
            }
        }
    }
}
