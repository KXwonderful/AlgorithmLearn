package array;

import java.util.Arrays;
import java.util.Comparator;

/**
 * 贪心算法：相比动态规划，使用贪心算法需要满足更多的条件（贪心选择性质），但效率比动态规划要高。
 * <p>
 * 贪心选择性质：每一步都做出一个局部最优的选择，最终的结果就是全局最优
 */
public class Greedy {

    /**
     * Interval Scheduling（区间调度问题）
     * <p>
     * 在闭区间 [start,end]，算出这些区间中最多有几个互不相交的区间。
     * <p>
     * 如 ints=[[1,3],[2,4],[3,6]]，区间最多有 2 个区间互不相交，即[[1,3],[3,6]]。
     */
    int intervalScheduling(int[][] ints) {
        // 1. 从区间集合 ints 中选择一个区间 x，这个 x 是在当前所有区间中结束最早的（end 最小）。
        // 2. 把所有与 x 区间相交的区间从区间集合 ints 中删除。
        // 3. 重复步骤 1 和 2，直到 ints 为空为止。之前选出的那些 x 就是最大不相交子集。

        if (ints.length == 0) return 0;
        // 按 end 升序排序
        Arrays.sort(ints, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });

        // 至少有一个区间不相交
        int count = 1;
        // 排序后，第一个区间就是 x
        int x_end = ints[0][1];
        for (int[] interval : ints) {
            int start = interval[0];
            if (start >= x_end) {
                // 找到下一个选择的区间了
                count++;
                x_end = interval[1];
            }
        }
        return count;
    }

    /**
     * 力扣 435 题，无重叠区间
     * <p>
     * 给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠
     */
    int eraseOverlapIntervals(int[][] intervals) {
        // 算出最多有几个区间不会重叠，那么剩下的就是至少需要去除的区间
        int n = intervals.length;
        return n - intervalScheduling(intervals);
    }

    /**
     * 力扣 452 题，用最少的箭头射爆气球
     * @param ints
     * @return
     */
    int findMinArrowShots(int[][] ints){
        // ...
        int count = 1;
        int x_end = ints[0][1];
        for (int[] interval : ints) {
            int start = interval[0];
            // 把 >= 改成 > 就行了
            if (start > x_end) {
                count++;
                x_end = interval[1];
            }
        }
        return count;
    }
}
