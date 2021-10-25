package number;

import java.util.LinkedList;

/**
 * 幂运算
 */
public class Exponentiation {

    public static void main(String[] args) {
        // test
        int a = 2;
        int[] b = new int[]{1, 0};
        // 答案：1024
        System.out.println("res=" + superPow(a, b));

    }

    /**
     * 372. 超级次方
     * <p>
     * 计算 a^b 对 1337 取模，a 是一个正整数，b 是一个非常大的正整数且会以数组形式给出。
     * <p>
     * 时间复杂度是 O(N)
     */
    public static int superPow(int a, int[] b) {
        LinkedList<Integer> list = new LinkedList<>();
        for (int i : b) {
            list.add(i);
        }
        return traverse(a, list);
    }

    private static int traverse(int a, LinkedList<Integer> b) {
        // 递归 base case
        if (b.isEmpty()) return 1;
        // 取出最后一个数并把它从链表移除
        int last = b.pollLast();
        // 将原问题化简，缩小规模递归求解
        int part1 = getPow(a, last);
        int part2 = getPow(traverse(a, b), 10);
        //

        return part1 * part2;
    }

    static int base = 1337;

    /**
     * 计算 a 的 k 次方的结果与 base 求模的结果
     * <p>
     * (a*b)%k = (a%k)(b%k)%k
     */
    private static int getPow(int a, int k) {
        // 对因子求模
        a %= base;
        int res = 1;
        for (int i = 0; i < k; i++) {
            // 这里有乘法，是潜在的溢出点
            res *= a;
            // 对乘法结果求模
            res %= base;
        }
        return res;
    }

    /**
     * 计算 a 的 k 次方的结果与 base 求模的结果
     */
    private static int getPow2(int a, int k) {
        if (k == 0) return 1;
        a %= base;

        if (k % 2 == 1) {
            // k 是奇数
            return (a * getPow2(a, k - 1)) % base;
        } else {
            // k 是偶数
            int sub = getPow2(a, k / 2);
            return (sub * sub) % base;
        }
    }
}
