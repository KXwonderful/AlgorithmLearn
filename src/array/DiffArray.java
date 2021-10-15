package array;

import java.util.Arrays;

/**
 * 差分数组：主要适用场景是频繁对原始数组的某个区间的元素进行增减
 */
public class DiffArray {

    // *****************************  差分数组 start  *****************************

    // diff[i]：nums[i] 和 nums[i-1] 之差
    // 差分数组
    private final int[] diff;

    /**
     * 输入一个数组，构造差分数组
     */
    public DiffArray(int[] nums) {
        diff = new int[nums.length];
        // 构造差分数组
        diff[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            diff[i] = nums[i] - nums[i - 1];
        }
    }

    /**
     * 给闭区间 [i,j] 增加 val（可以是负数）
     * <p>
     * diff[i] += val 意味着给 nums[i..] 所有的元素都加了 val，
     * diff[j+1] -= val 意味着对于 nums[j+1..] 所有元素再减 val，
     * 最终对 nums[i..j] 中的所有元素都加 val
     */
    public void increment(int i, int j, int val) {
        diff[i] += val;
        if (j + 1 < diff.length) { // 当j+1 >= diff.length时，说明对nums[i]及后面整个数组都进行修改，就不需要再给diff数组减val了
            diff[j + 1] -= val;
        }
    }

    /**
     * 通过 diff 差分数组是反推出原始数组 nums
     */
    public int[] result() {
        System.out.println("diff=" + Arrays.toString(diff));
        int[] res = new int[diff.length];
        // 根据差分数组构造结果数组
        res[0] = diff[0];
        for (int i = 1; i < diff.length; i++) {
            res[i] = res[i - 1] + diff[i];
        }
        return res;
    }

    // *****************************   差分数组 end   *****************************


    public static void main(String[] args) {
        // test
        int[][] bookings = new int[][]{{1, 2, 10}, {2, 3, 20}, {2, 5, 25}};
        int n = 5;
        int[] res = corpFlightBookings(bookings, n);
        // 答案：[10,55,45,25,25]
        System.out.println("res=" + Arrays.toString(res));
    }


    /**
     * 力扣 1109. 航班预订统计
     */
    public static int[] corpFlightBookings(int[][] bookings, int n) {
        // nums 初始化为全 0
        int[] nums = new int[n];
        //  构造差分解法
        DiffArray diff = new DiffArray(nums);

        for (int[] booking : bookings) {
            // 注意转成数组索引要减一
            int i = booking[0] - 1;
            int j = booking[1] - 1;
            int val = booking[2];
            // 对区间 nums[i..j] 增加 val
            diff.increment(i, j, val);
        }
        // 返回最终的结果数组
        return diff.result();
    }

}
