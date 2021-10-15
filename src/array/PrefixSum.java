package array;

import java.util.HashMap;

/**
 * 前缀和：主要适用的场景是原始数组不会被修改的情况下，频繁查询某个区间的累加和
 */
public class PrefixSum {

    // *****************************  前缀和 start  *****************************
    // 前缀和数组
    private final int[] prefix;

    /**
     * 输入一个数组，构造前缀和
     */
    public PrefixSum(int[] nums) {
        prefix = new int[nums.length + 1];
        // 计算 nums 的累加和
        for (int i = 0; i < prefix.length; i++) {
            prefix[i + 1] = prefix[i] + nums[i];
        }
    }

    /**
     * 查询闭区间 [i, j] 的累加和
     */
    public int query(int i, int j) {
        return prefix[j + 1] - prefix[i];
    }

    // *****************************   前缀和 end   *****************************

    public static void main(String[] args) {
        // test1
        int[] nums1 = new int[]{1, 1, 1};
        int k1 = 2;
        int res1 = subArraySum(nums1, k1);
        System.out.println("res1 = " + res1);

        // test2
        int[] nums2 = new int[]{1, 2, 3};
        int k2 = 3;
        int res2 = subArraySum(nums2, k2);
        System.out.println("res2 = " + res2);
    }

    /**
     * 剑指 Offer II 010. 和为 k 的子数组
     * <p>
     * 给定一个整数数组和一个整数 k ，请找到该数组中和为 k 的连续子数组的个数。
     * <p>
     * 时间复杂度 O(N^2) 空间复杂度 O(N)
     */
    private static int subArraySum(int[] nums, int k) {
        int n = nums.length;
        // 构造前缀和
        int[] sum = new int[n + 1];
        sum[0] = 0;
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + nums[i];
        }

        int ans = 0;
        // 穷举所有子数组
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                // sum of nums[j..i-1]
                if (sum[i] - sum[j] == k) ans++;
            }
        }

        return ans;
    }

    /**
     * 优化解法：记录下有几个sum[j]和sum[i]-k相等，直接更新结果，就避免了内层的 for 循环
     * <p>
     * 判断条件 if (sum[i] - sum[j] == k) ans++ 换成 if (sum[j] == sum[i] - k) ans++
     * <p>
     * 时间复杂度 O(N)
     */
    private static int subArraySum2(int[] nums, int k) {
        // map: 前缀和 -> 该前缀和出现的次数
        HashMap<Integer, Integer> preSum = new HashMap<>();
        // base case
        preSum.put(0, 1);

        int ans = 0, sum0_i = 0;
        for (int num : nums) {
            sum0_i += num;
            // 要找的前缀和 nums[0..j]
            int sum0_j = sum0_i - k;
            // 若前面有这个前缀和，直接更新答案
            if (preSum.containsKey(sum0_j)) ans += preSum.get(sum0_j);
            // 把前缀和 nums[0..i] 加入并记录出现的次数
            preSum.put(sum0_i, preSum.getOrDefault(sum0_i, 0) + 1);
        }

        return ans;
    }

}
