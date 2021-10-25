package number;

import java.util.Arrays;

/**
 * 素数：如果一个数如果只能被 1 和它本身整除，那么这个数就是素数。
 */
public class Prime {

    public static void main(String[] args) {
        // test
        int n = 10;
        // 答案：4
        System.out.println("res=" + countPrimes(n));
        System.out.println("res=" + countPrimes2(n));
        System.out.println("res=" + countPrimes3(n));
    }

    /**
     * 返回区间 [2, n) 中有几个素数
     * <p>
     * 比如 countPrimes(10) 返回 4，因为 2,3,5,7 是素数
     * <p>
     * 时间复杂度 O(n^2)
     */
    public static int countPrimes(int n) {
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrime(i)) count++;
        }
        return count;
    }

    /**
     * 判断整数 n 是否是素数
     */
    private static boolean isPrime(int n) {
        for (int i = 2; i < n; i++) {
            if (n % i == 0) {
                // 有其他整除因子
                return false;
            }
        }
        return true;
    }

    /**
     * 判断整数 n 是否是素数：优化
     * <p>
     * isPrime函数的时间复杂度降为了 O(sqrt(N))
     */
    private static boolean isPrime2(int n) {
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }


    // *****************************   高效实现 countPrimes   *****************************

    public static int countPrimes2(int n) {
        boolean[] isPrim = new boolean[n];
        // 将数组都初始化为 true
        Arrays.fill(isPrim, true);

        for (int i = 2; i < n; i++) {
            if (isPrim[i]) {
                // i 的倍数不可能是素数了
                for (int j = 2 * i; j < n; j += i) {
                    isPrim[j] = false;
                }
            }
        }

        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrim[i]) count++;
        }
        return count;
    }

    /**
     * 时间复杂度 O(N * loglogN)
     */
    public static int countPrimes3(int n) {
        boolean[] isPrim = new boolean[n];
        // 将数组都初始化为 true
        Arrays.fill(isPrim, true);

        for (int i = 2; i * i < n; i++) {
            if (isPrim[i]) {
                // i 的倍数不可能是素数了
                for (int j = i * i; j < n; j += i) {
                    isPrim[j] = false;
                }
            }
        }

        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrim[i]) count++;
        }
        return count;
    }
}
