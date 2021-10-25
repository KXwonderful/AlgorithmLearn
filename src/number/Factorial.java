package number;

/**
 * 阶乘
 */
public class Factorial {

    public static void main(String[] args) {
        // test
        int n = 5;
        int k = 1;
        // 答案：1，5
        long timeStart = System.currentTimeMillis();
        int res1 = trailingZeroes(n);
        long timeEnd = System.currentTimeMillis();
        long timeConsuming1 = timeEnd - timeStart;

        long res2 = preimageSizeFZF(k);
        long timeEnd2 = System.currentTimeMillis();
        long timeConsuming2 = timeEnd2 - timeEnd;

        System.out.println("trailingZeroes res=" + res1 + " 耗时：" + timeConsuming1);
        System.out.println("preimageSizeFZF res=" + res2 + " 耗时：" + timeConsuming2);
    }

    /**
     * 输入一个非负整数 n，请你计算阶乘 n! 的结果末尾有几个 0
     * <p>
     * 输入 n = 5，算法返回 1，因为 5! = 120，末尾有一个 0。
     */
    public static int trailingZeroes(int n) {
        // 不能直接把n!的结果算出来，阶乘增长比指数增长还要大

        // 两个数相乘结果末尾有 0，一定是因为两个数中有因子 2 和 5，因为 10 = 2 x 5
        // 思路：问题转化为：n!最多可以分解出多少个因子 2 和 5？

        // 如 n = 25，那么25!最多可以分解出几个 2 和 5 相乘？
        // 这个主要取决于能分解出几个因子 5，因为每个偶数都能分解出因子 2，因子 2 肯定比因子 5 多得多。
        // 思路：问题进一步转化为：n!最多可以分解出多少个因子 5？

        int res = 0;
        long divisor = 5;
        while (divisor <= n) {
            res += n / divisor;
            divisor *= 5;
        }
        return res;
    }

    /**
     * 时间复杂度 O(logN)
     */
    public static int trailingZeroes2(int n) {
        int res = 0;
        for (int d = n; d / 5 > 0; d = d / 5) {
            res += d / 5;
        }
        return res;
    }

    /**
     * 输入一个非负整数 K，请你计算有多少个 n，满足 n! 的结果末尾恰好有 K 个 0
     * <p>
     * 输入 K = 1，算法返回 5，因为 5!,6!,7!,8!,9! 这 5 个阶乘的结果最后只有一个 0，即有 5 个 n 满足条件
     * <p>
     * 时间复杂度 O(logN*logN)
     */
    public static long preimageSizeFZF(int K) {
        // 对于这种具有单调性的函数，用 for 循环遍历，可以用二分查找进行降维打击

        // 搜索有多少个n满足trailingZeroes(n) == K，其实就是在问，
        // 满足条件的n最小是多少，最大是多少，最大值和最小值一减，就可以算出来有多少个n满足条件了

        // n属于区间[0,LONG_MAX]，我们要寻找满足trailingZeroes(n) == K的左侧边界和右侧边界

        // 左边界和右边界之差 + 1 就是答案
        return  right_bound(K) - left_bound(K) + 1;
    }

    /**
     * 搜索 trailingZeroes(n) == K 的左侧边界
     */
    private static long left_bound(int target) {
        long low = 0, high = Long.MAX_VALUE;
        while (low < high) {
            long mid = low + (high - low) / 2;
            if (trailingZeroes(mid) < target) {
                low = mid + 1;
            } else if (trailingZeroes(mid) > target) {
                high = mid;
            } else {
                high = mid;
            }
        }

        return low;
    }

    /**
     * 搜索 trailingZeroes(n) == K 的右侧边界
     */
    private static long right_bound(int target) {
        long low = 0, high = Long.MAX_VALUE;
        while (low < high) {
            long mid = low + (high - low) / 2;
            if (trailingZeroes(mid) < target) {
                low = mid + 1;
            } else if (trailingZeroes(mid) > target) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        return low - 1;
    }

    /**
     * 注意为了避免整型溢出的问题，trailingZeroes函数需要把所有数据类型改成 long
     * <p>
     * 逻辑不变，数据类型全部改成 long
     */
    private static long trailingZeroes(long n) {
        long res = 0;
        for (long d = n; d / 5 > 0; d = d / 5) {
            res += d / 5;
        }
        return res;
    }
}
