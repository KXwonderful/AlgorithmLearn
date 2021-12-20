package other;

import java.util.HashMap;

public class DynamicTest {

    public static void main(String[] args) {

    }

    /**
     * 零钱兑换 -- 备忘录
     */
    public int coinChange(int[] coins, int amount) {
        return dp(coins, amount);
    }

    HashMap<Integer, Integer> memo = new HashMap<>();

    // 要凑出金额 n，至少要 dp(n) 个硬币
    int dp(int[] coins, int n) {
        if (memo.containsKey(n)) return memo.get(n);
        if (n == 0) return 0;
        if (n < 0) return -1;

        int res = Integer.MAX_VALUE;
        for (int coin : coins) {
            int child = dp(coins, n - coin);
            if (child == -1) continue;

            res = Math.min(res, child + 1);
        }

        memo.put(n, res == Integer.MAX_VALUE ? -1 : res);

        return memo.get(n);
    }

    /**
     * 零钱兑换: -- dp table
     * <p>
     * dp[i] = x 表示，当目标金额为 i 时，至少需要 x 枚硬币
     */
    public int coinChange2(int[] coins, int amount) {

        int[] dp = new int[amount + 1];
        for (int i = 0; i <= amount; i++) {
            dp[i] = amount + 1;
        }

        // base case
        dp[0] = 0;
        for (int i = 0; i < dp.length; i++) {
            for (int coin : coins) {
                if (i - coin < 0) continue;

                dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
            }
        }

        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    /**
     * 路径最小和
     */
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        int[][] memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                memo[i][j] = -1; // 初始值全部设为 -1
            }
        }

        return dp(memo, grid, m - 1, n - 1);
    }


    // dp 函数：从左上角位置 (0, 0) 走到位置 (i, j) 的最小路径和为 dp(grid, i, j)
    int dp(int[][] memo, int[][] grid, int i, int j) {
        if (i < 0 || j < 0) return Integer.MAX_VALUE;

        // base case
        if (i == 0 && j == 0) return grid[i][j];

        if (memo[i][j] != -1) return memo[i][j];

        // 左边和上面的最小路径和加上 grid[i][j] 就是到达 (i, j) 的最小路径和
        memo[i][j] = grid[i][j] + Math.min(dp(memo, grid, i - 1, j), dp(memo, grid, i, j - 1));
        return memo[i][j];
    }

    /**
     * 剪绳子：
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。
     * 请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
     */
    public int cuttingRope(int n) {

        // dp 数组：dp[i]表示长度为i的绳子剪成m端后长度的最大乘积(m>1)
        int[] dp = new int[n + 1];

        // base case
        dp[2] = 1;
        // 目标：求出dp[n]
        for (int i = 3; i <= n; i++) {
            // 首先对绳子剪长度为j的一段,其中取值范围为 2 <= j < i
            for (int j = 2; j < i; j++) {
                // Math.max(j*(i-j),j*dp[i-j]是由于减去第一段长度为j的绳子后，可以继续剪也可以不剪
                dp[i] = Math.max(dp[i], Math.max(j * (i - j), j * dp[i - j]));
            }
        }
        return dp[n];
    }

}
