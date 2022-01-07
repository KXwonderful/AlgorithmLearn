import java.util.*;

public class Testing {

    public static void main(String[] args) {

    }

    private ListNode traverse1(ListNode head) {
        if (head.next == null) return head;
        ListNode last = traverse1(head.next);
        head.next.next = head;
        head.next = null;
        return last;
    }


    ListNode flag = null;

    private ListNode traverse2(ListNode head, int n) {
        if (n == 1) {
            flag = head.next;
            return head;
        }
        ListNode last = traverse2(head.next, n - 1);
        head.next.next = head;
        head.next = flag;


        return last;
    }

    private ListNode traverse3(ListNode head, int m, int n) {
        if (m == 1) {
            return traverse2(head, n);
        }
        head.next = traverse3(head, m - 1, n - 1);
        return head;
    }

    private void insertSort(int[] array) {
        for (int i = 1; i < array.length; i++) {
            int temp = array[i];// 记录要插入的数据
            // 从已经排序的序列最右边的开始比较，找到比其小的数
            int j = i;
            while (j > 0 && array[j - 1] > temp) {
                array[j] = array[j - 1];
                j--;
            }
            if (j != i) array[j] = temp;
        }
    }

    /**
     * 151. 翻转字符串里的单词
     */
    public String reverseWords(String s) {
        String[] words = s.trim().split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            if (!words[i].isEmpty()) {
                sb.append(words[i]).append(" ");
            }
        }
        return sb.toString().trim();
    }

    /**
     * 54. 螺旋矩阵：给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序，返回矩阵中的所有元素。
     * <p>
     * 按层模拟遍历
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0) return res;

        int top = 0, left = 0, bottom = matrix.length - 1, right = matrix[0].length - 1;

        while (left <= right && top <= bottom) {
            //1 从(top, left) -> (top, right)
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }
            //2 从(top + 1, right) -> (bottom, right)
            for (int i = top + 1; i <= bottom; i++) {
                res.add(matrix[i][right]);
            }

            if (top < bottom && left < right) {
                //3 从(bottom, right - 1) -> (bottom, left + 1)
                for (int i = right - 1; i > left; i--) {
                    res.add(matrix[bottom][i]);
                }
                //4 从(bottom, left) -> (top - 1, left)
                for (int i = bottom; i > top; i--) {
                    res.add(matrix[i][left]);
                }
            }

            left++;
            top++;
            right--;
            bottom--;
        }

        return res;
    }

    /**
     * 剑指 Offer 61. 扑克牌中的顺子
     */
    public boolean isStraight(int[] nums) {
        Set<Integer> repeat = new HashSet<>();
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        for (int num : nums) {
            if (num == 0) continue;
            max = Math.max(max, num);
            min = Math.min(min, num);
            if (repeat.contains(num)) return false;
            repeat.add(num);
        }
        return max - min < 5;
    }

    /**
     * 剑指 Offer 62. 圆圈中最后剩下的数字
     */
    public int lastRemaining(int n, int m) {
        int res = 0;
        // 最后一轮剩下2个人，所以从2开始反推
        for (int i = 2; i <= n; i++) {
            res = (res + m) % i;
        }
        return res;
    }

    /**
     * 剑指 Offer 65. 不用加减乘除做加法
     */
    public int add(int a, int b) {
        while (b != 0) {// 当进位为 0 时跳出
            int c = (a & b) << 1; // c = 进位
            a ^= b; // a = 非进位和
            b = c; // b = 进位
        }
        return a;
    }

    public int add2(int a, int b) {
        if (b == 0) {
            return a;
        }
        return add2(a ^ b, (a & b) << 1);
    }

    /**
     * 剑指 Offer 45. 把数组排成最小的数
     */
    public String minNumber(int[] nums) {
        int len = nums.length;
        String[] strs = new String[len];
        for (int i = 0; i < len; i++) {
            strs[i] = String.valueOf(nums[i]);
        }
        // 使用内置函数进行排序（或用其他排序算法）
        Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x));
        StringBuilder sb = new StringBuilder();
        for (String str : strs) {
            sb.append(str);
        }
        return sb.toString();
    }

    /**
     * 剑指 Offer 46. 把数字翻译成字符串
     * 时间复杂度 O(N), 空间复杂度 O(N)
     */
    public int translateNum(int num) {
        // dp[i] 代表以 i 为结尾的数字的翻译方案数量。
        // 区间 [10, 25]：dp[i] = dp[i-1] + dp[i-2]
        // 区间 [0, 10) V (25, 99]：dp[i] = dp[i-1]
        String s = String.valueOf(num);
        int len = s.length();
        int[] dp = new int[len + 1];
        // base case
        dp[0] = dp[1] = 1; // 即 “无数字” 和 “第 1 位数字” 的翻译方法数量均为 1
        for (int i = 2; i <= len; i++) {
            String temp = s.substring(i - 2, i);
            dp[i] = temp.compareTo("10") >= 0 && temp.compareTo("25") <= 0 ? dp[i - 1] + dp[i - 2] : dp[i - 1];
        }
        return dp[len];
    }

    // 优化：
    public int translateNum2(int num) {
        // dp[i] 代表以 i 为结尾的数字的翻译方案数量。
        // 区间 [10, 25]：dp[i] = dp[i-1] + dp[i-2]
        // 区间 [0, 10) V (25, 99]：dp[i] = dp[i-1]
        String s = String.valueOf(num);
        int len = s.length();
        // base case
        int a = 1, b = 1; // 即 “无数字” 和 “第 1 位数字” 的翻译方法数量均为 1
        for (int i = 2; i <= len; i++) {
            String temp = s.substring(i - 2, i);
            int c = temp.compareTo("10") >= 0 && temp.compareTo("25") <= 0 ? a + b : a;
            b = a;
            a = c;
        }
        return a;
    }

    /**
     * 剑指 Offer 47. 礼物的最大价值（路径最大和）
     */
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        // 备忘录
        memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                memo[i][j] = -1;
            }
        }

        return dp(grid, m - 1, n - 1);
    }

    int[][] memo;

    // 从左上角位置 (0, 0) 走到位置 (i, j) 的最大路径和为 dp(grid, i, j)
    int dp(int[][] grid, int i, int j) {
        if (i < 0 || j < 0) return Integer.MIN_VALUE;

        // base case
        if (i == 0 && j == 0) return grid[i][j];

        // 查找备忘录
        if (memo[i][j] != -1) return memo[i][j];

        // 添加到备忘录
        memo[i][j] = grid[i][j] + Math.max(dp(grid, i - 1, j), dp(grid, i, j - 1));

        return memo[i][j];
    }

    /**
     * 剑指 Offer 48. 最长不含重复字符的子字符串
     */
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> window = new HashMap<>();

        int left = 0, right = 0;
        int res = 0;

        while (right < s.length()) {
            char c = s.charAt(right);
            // 右移窗口
            right++;
            // 进行窗口内数据的一系列更新
            window.put(c, window.getOrDefault(c, 0) + 1);

            // 判断窗口是否要收缩
            while (window.get(c) > 1) {
                char d = s.charAt(left);
                // 左移窗口
                left++;
                // 进行窗口内数据的一系列更新
                window.put(d, window.getOrDefault(d, 0) - 1);
            }

            res = Math.max(res, right - left);
        }

        return res;
    }

    /**
     * 剑指 Offer 49. 丑数
     * <p>
     * 给你一个整数 n ，请你找出并返回第 n 个 丑数。
     * 把只包含质因子 2、3 和 5 的数称作丑数，（注：1 视为丑数）
     */
    public int nthUglyNumber(int n) {
        // dp[i] 代表第 i + 1 个丑数
        int[] dp = new int[n];

        // base case
        dp[0] = 1;

        // 根据丑数性质：
        // dp[i] 是最接近 dp[i-1] 的丑数，索引 a,b,c 需满足如下条件
        // dp[a]×2 > dp[i−1] ≥ dp[a−1]×2
        // dp[b]×3 > dp[i−1] ≥ dp[b−1]×3
        // dp[c]×5 > dp[i−1] ≥ dp[c−1]×5
        int a = 0, b = 0, c = 0; // 初始化索引
        for (int i = 1; i < n; i++) {
            int n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5;
            dp[i] = Math.min(Math.min(n2, n3), n5);
            if (dp[i] == n2) a++;
            if (dp[i] == n3) b++;
            if (dp[i] == n5) c++;
        }

        // dp[n−1] ，即返回第 n 个丑数
        return dp[n - 1];
    }

    /**
     * 剑指 Offer 56 - I. 数组中数字出现的次数
     * <p>
     * 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次，找出这两个只出现一次的数字。
     * <p>
     * 要求时间复杂度是O(n)，空间复杂度是O(1)
     */
    public int[] singleNumbers(int[] nums) {
        int x = 0, y = 0, n = 0, m = 1;
        // 1. 遍历异或
        for (int num : nums) {
            n ^= num;
        }
        // 2. 循环左移，计算 m
        while ((n & m) == 0) {
            m <<= 1;
        }
        // 3. 遍历 nums 分组
        for (int num : nums) {
            if ((num & m) != 0) {
                x ^= num;
            } else {
                y ^= num;
            }
        }
        return new int[]{x, y};
    }

    /**
     * 剑指 Offer 56 - II. 数组中数字出现的次数 II
     * <p>
     * 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
     */
    public int singleNumber2(int[] nums) {
        int ones = 0, twos = 0;
        for (int num : nums) {
            ones = ones ^ num & ~twos;
            twos = twos ^ num & ~ones;
        }
        return ones;
    }

    /**
     * 136. 只出现一次的数字
     * <p>
     * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
     */
    public int singleNumber(int[] nums) {
        // 一个数和它本身做异或运算结果为 0，即 a ^ a = 0；
        // 一个数和 0 做异或运算的结果为它本身，即 a ^ 0 = a
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }
        return res;
    }

    /**
     * 剑指 Offer 31. 栈的压入、弹出序列
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        // 利用栈的先进后出特性
        Stack<Integer> stack = new Stack<>();
        int index = 0;
        for (int n : pushed) {
            // 压栈
            stack.push(n);
            while (!stack.isEmpty() && stack.peek() == popped[index]) {
                // 出栈
                stack.pop();
                index++;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 剑指 Offer 30. 包含min函数的栈
     * <p>
     * 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
     */
    class MinStack {

        Stack<Integer> s1, s2;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            s1 = new Stack<>(); // 数据栈
            s2 = new Stack<>(); // 辅助栈，存储s1中非严格降序元素，把最小元素存入栈顶
        }

        // 入栈
        public void push(int x) {
            s1.add(x);
            if (s2.isEmpty() || s2.peek() >= x) {
                // s2为空或 x 小等于栈顶元素
                s2.add(x);
            }
        }

        // 出栈
        public void pop() {
            int y = s1.pop();
            if (y == s2.peek()) {
                // 若 y 等于栈顶元素则出栈
                s2.pop();
            }
        }

        // 栈顶元素
        public int top() {
            return s1.peek();
        }

        // 最小值
        public int min() {
            return s2.peek();
        }
    }

    /**
     * 剑指 Offer 59 - II. 队列的最大值
     * <p>
     * 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。
     */
    class MaxQueue {

        Queue<Integer> q1;
        Deque<Integer> q2; // 双向队列

        public MaxQueue() {
            q1 = new LinkedList<>();
            q2 = new LinkedList<>();
        }

        // 最大值
        public int max_value() {
            return q2.isEmpty() ? -1 : q2.peekFirst();
        }

        // 入队
        public void push_back(int value) {
            q1.offer(value);
            // 将双向队列中队尾所有小于 value 的元素弹出（以保持 deque 非单调递减），并将元素 value 入队 deque
            while (!q2.isEmpty() && q2.peekLast() < value) {
                q2.pollLast();
            }
            q2.offerLast(value);
        }

        // 出队
        public int pop_front() {
            if (q1.isEmpty()) {
                return -1;
            }
            int pop = q1.poll();
            if (!q2.isEmpty() && pop == q2.peekFirst()) {
                q2.pollFirst();
            }
            return pop;
        }
    }


    /**
     * 剑指 Offer 60. n个骰子的点数
     * <p>
     * 把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
     */
    public double[] dicesProbability(int n) {

        // 设输入 n 个骰子的解（即概率列表）为 f(n)，其中「点数和」 x 的概率为 f(n, x)
        double[] dp = new double[6];
        Arrays.fill(dp, 1.0 / 6.0);
        for (int i = 2; i <= n; i++) {
            double[] temp = new double[5 * i + 1]; // i个骰子点数之和的个数为 6*i-(i-1)，化简：5*i+1

            // 从i-1个骰子的点数之和的值数组入手，计算i个骰子的点数之和数组的值
            // 先拿i-1个骰子的点数之和数组的第j个值，它所影响的是i个骰子时的temp[j+k]的值
            for (int j = 0; j < dp.length; j++) {
                for (int k = 0; k < 6; k++) {
                    // 这里加上dp数组值与1/6的乘积，1/6是第i个骰子投出某个值的概率
                    temp[j + k] += dp[j] / 6.0;
                }
            }
            dp = temp;
        }
        return dp;
    }

    /**
     * 剑指 Offer 64. 求1+2+…+n
     * <p>
     * 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
     */
    public int sumNums(int n) {
        // return (1 + n) * n / 2;

        // if(A && B)  若 A 为 false ，则 B 的判断不会执行（即短路），直接判定 A && B 为 false
        // if(A || B)  若 A 为 true ，则 B 的判断不会执行（即短路），直接判定 A || B 为 true

        // 要实现 “当 n = 1 时终止递归”
        // n > 1 && sumNums(n - 1) // 当 n = 1 时 n > 1 不成立 ，此时 “短路” ，终止后续递归
        boolean x = n > 1 && (n += sumNums(n - 1)) > 1;
        return n;
    }

    /**
     * 剑指 Offer 66. 构建乘积数组
     * <p>
     * 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中B[i] 的值是数组 A 中除了下标 i 以外的元素的积,
     * 即B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
     */
    public int[] constructArr(int[] a) {
        int len = a.length;
        if (len == 0) return new int[0];

        int[] b = new int[len];
        b[0] = 1;
        int temp = 1;
        // 计算左下三角
        for (int i = 1; i < len; i++) {
            b[i] = b[i - 1] * a[i - 1];
        }
        // 计算右上三角
        for (int i = len - 2; i >= 0; i--) {
            temp *= a[i + 1];
            b[i] *= temp;
        }
        return b;
    }

    /**
     * 剑指 Offer 67. 把字符串转换成整数
     */
    public int strToInt(String str) {
        double res = Double.parseDouble(str);
        if (res > Integer.MAX_VALUE) {

        }
        StringBuilder sb = new StringBuilder();

        return (int) res;
    }


    /**
     * 单链表节点
     */
    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int val) {
            this.val = val;
        }
    }
}
