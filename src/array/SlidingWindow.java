package array;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * 滑动窗口
 */
public class SlidingWindow {

    /**
     * 滑动窗口框架
     */
    public void slidingWindow(String s, String t) {
        HashMap<Character, Integer> need = new HashMap<>(), window;
        for (Character c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }

        int left = 0, right = 0;
        int valid = 0;
        while (right < s.length()) {
            char c = s.charAt(right);
            // 右移窗口
            right++;
            // 进行窗口内数据的一系列更新
            // todo right sth

            // 判断窗口是否要收缩
            while (windowNeedsShrink()) {
                char d = s.charAt(left);
                // 左移窗口
                left++;
                // 进行窗口内数据的一系列更新
                // todo left sth
            }
        }
    }

    boolean windowNeedsShrink() {
        return true;
    }


    /**
     * 76. 最小覆盖子串
     */
    public String minWindow(String s, String t) {
        HashMap<Character, Integer> need = new HashMap<>(), window = new HashMap<>();
        for (Character c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }

        int left = 0, right = 0;
        int valid = 0; // 窗口中满足 need 条件的字符个数
        int start = 0, len = Integer.MAX_VALUE; // 记录最小覆盖子串的起始索引及长度
        while (right < s.length()) {
            char c = s.charAt(right);

            // 右移窗口
            right++;
            if (need.containsKey(c)) {
                // 存入window
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }

            // 判断窗口是否要收缩
            while (valid == need.size()) {
                if (right - left < len) {
                    start = left;
                    len = right - left;
                }

                char d = s.charAt(left);
                // 左移窗口
                left++;
                if (need.containsKey(d)) {
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    window.put(d, window.get(d) - 1);
                }
            }
        }

        return len == Integer.MAX_VALUE ? "" : s.substring(start, start + len);
    }


    /**
     * 567. 字符串的排列: 判断 s 中是否存在 t 的排列
     */
    public boolean checkInclusion(String s, String t) {
        if (t.length() > s.length()) return false;

        HashMap<Character, Integer> need = new HashMap<>(), window = new HashMap<>();
        for (Character c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }

        int left = 0, right = 0;
        int valid = 0;

        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            if (need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }

            while (right - left >= t.length()) {

                if (valid == need.size()) return true;

                char d = s.charAt(left);
                left++;
                if (need.containsKey(d)) {
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    window.put(d, window.getOrDefault(d, 0) - 1);
                }
            }
        }

        return false;
    }

    /**
     * 438.  找到字符串中所有字母异位词
     */
    public List<Integer> findAnagrams(String s, String p) {
        HashMap<Character, Integer> need = new HashMap<>(), window = new HashMap<>();
        for (Character c : p.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }

        int left = 0, right = 0;
        int valid = 0;
        List<Integer> res = new ArrayList<>();

        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            if (need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }

            while (right - left >= p.length()) {
                if (valid == need.size()) {
                    res.add(left);
                }

                char d = s.charAt(left);
                left++;
                if (need.containsKey(d)) {
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    window.put(d, window.getOrDefault(d, 0) - 1);
                }
            }

        }
        return res;
    }

    /**
     * 3. 无重复字符的最长子串
     */
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> window = new HashMap<>();

        int left = 0, right = 0;
        int res = 0;

        while (right < s.length()) {
            char c = s.charAt(right);
            right++;

            window.put(c, window.getOrDefault(c, 0) + 1);

            while (window.get(c) > 1) {
                char d = s.charAt(left);
                left++;
                window.put(d, window.getOrDefault(d, 0) - 1);
            }

            res = Math.max(res, right - left);
        }

        return res;
    }

    /**
     * 和为s的连续正数序列
     */
    public int[][] findContinuousSequence(int target) {
        int left = 1, right = 1;
        int sum = 0;
        List<int[]> res = new ArrayList<>();

        while (left <= target / 2) {
            if (sum < target) {
                sum += right;
                right++;
            } else if (sum > target) {
                sum -= left;
                left++;
            } else {

                int[] arr = new int[right - left];
                for (int i = left; i < right; i++) {
                    arr[i - left] = i;
                }

                res.add(arr);
                sum -= left;
                left++;
            }
        }
        return res.toArray(new int[res.size()][]);
    }
}
