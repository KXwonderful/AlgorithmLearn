import java.util.*;

public class Testing {

    public static void main(String[] args) {
        int[] arr = new int[]{-15, -5, 1, 1, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 8, 8, 10, 13};
        List<List<Integer>> res = findTwoArray(arr, 9);
        for (List<Integer> list : res) {
            System.out.println(list);
        }
        System.out.println("================");
        List<List<Integer>> res2 = findThreeArray(arr, 9);
        for (List<Integer> list : res2) {
            System.out.println(list);
        }
        System.out.println("================");
        List<List<Integer>> res3 = twoSumTarget(arr, 0, 9);
        for (List<Integer> list : res3) {
            System.out.println(list);
        }
        System.out.println("================");
        List<List<Integer>> res4 = threeSumTarget(arr, 9);
        for (List<Integer> list : res4) {
            System.out.println(list);
        }
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
     * 剑指 Offer 19. 正则表达式匹配
     * <p>
     * 给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
     * '.' 匹配任意单个字符
     * '*' 匹配零个或多个前面的那一个元素
     */
    public boolean isMatch(String s, String p) {
//        // 看两个字符是否匹配，一切逻辑围绕匹配/不匹配两种情况展开即可
//        int index_s = 0, index_p = 0;
//        while (index_s < s.length() && index_p < p.length()) {
//            // 「.」通配符就是万金油
//            if (s.charAt(index_s) == p.charAt(index_p) || p.charAt(index_p) == '.') {
//                // 匹配
//                if (index_p < p.length() - 1 && p.charAt(index_p + 1) == '*') {
//                    // 有 * 通配符，可以匹配 0 次或多次
//                } else {
//                    // 无 * 通配符，老老实实匹配 1 次
//                    index_s++;
//                    index_p++;
//                }
//            } else {
//                // 不匹配
//                if (index_p < p.length() - 1 && p.charAt(index_p + 1) == '*') {
//                    // 有 * 通配符，只能匹配 0 次
//                } else {
//                    // 无 * 通配符，匹配无法进行下去了
//                    return false;
//                }
//            }
//        }
//
//        return index_s == index_p;
        memoMatch = new HashMap<>();
        return dp(s, 0, p, 0);
    }

    HashMap<String, Boolean> memoMatch;

    /**
     * dp(s,i,p,j) = true：表示 s[i..] 可以匹配 p[j..]
     * dp(s,i,p,j) = false：则表示 s[i..] 无法匹配 p[j..]
     */
    boolean dp(String s, int i, String p, int j) {

        // base case
        if (j == p.length()) return i == s.length();
        if (i == s.length()) {
            // 如果能匹配空串，一定是字符和 * 成对儿出现
            if ((p.length() - j) % 2 == 1) return false;
            // 检查是否为 x*y*z* 这种形式
            for (; j + 1 < p.length(); j += 2) {
                if (p.charAt(j + 1) != '*') return false;
            }
            return true;
        }

        // 记录状态 (i, j)，消除重叠子问题
        String key = i + "," + j;
        if (memoMatch.containsKey(key)) return memoMatch.get(key);
        boolean res;

        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            // 匹配
            if (j < p.length() - 1 && p.charAt(j + 1) == '*') {
                // 1.1 通配符匹配 0 次 或多次
                res = dp(s, i, p, j + 2) || dp(s, i + 1, p, j);
            } else {
                // 1.2 常规匹配 1 次
                res = dp(s, i + 1, p, j + 1);
            }
        } else {
            // 不匹配
            if (j < p.length() - 1 && p.charAt(j + 1) == '*') {
                // 2.1 通配符匹配 0 次
                res = dp(s, i, p, j + 2);
            } else {
                // 2.2 无法继续匹配
                res = false;
            }
        }

        // 添加备忘录
        memoMatch.put(key, res);
        return res;
    }

    /**
     * 剑指 Offer 37. 序列化二叉树 -- 前序遍历
     */
    public static class Codec {

        String SEP = ",";
        String NULL = "#";


        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serialize(root, sb);
            return sb.toString();
        }

        // 将二叉树打平为字符串 -- 前序遍历
        void serialize(TreeNode root, StringBuilder sb) {
            if (root == null) {
                sb.append(NULL).append(SEP);
                return;
            }

            sb.append(root.val).append(SEP);

            serialize(root.left, sb);
            serialize(root.right, sb);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            LinkedList<String> nodes = new LinkedList<>();
            for (String s : data.split(SEP)) {
                nodes.addLast(s);
            }
            return deserialize(nodes);
        }

        // 通过 nodes 列表构造二叉树 -- 前序遍历
        TreeNode deserialize(LinkedList<String> nodes) {
            if (nodes.isEmpty()) return null;

            // 前序遍历得到的 nodes 列表中，第一个元素是 root 节点的值
            String first = nodes.removeFirst();
            if (first.equals(NULL)) return null;
            TreeNode root = new TreeNode(Integer.parseInt(first));

            root.left = deserialize(nodes);
            root.right = deserialize(nodes);
            return root;
        }
    }

    /**
     * 剑指 Offer 37. 序列化二叉树 -- 后序遍历
     */
    public static class Codec2 {

        String SEP = ",";
        String NULL = "#";


        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serialize(root, sb);
            return sb.toString();
        }

        // 将二叉树打平为字符串 -- 后序遍历
        void serialize(TreeNode root, StringBuilder sb) {
            if (root == null) {
                sb.append(NULL).append(SEP);
                return;
            }
            serialize(root.left, sb);
            serialize(root.right, sb);

            sb.append(root.val).append(SEP);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            LinkedList<String> nodes = new LinkedList<>();
            for (String s : data.split(SEP)) {
                nodes.addLast(s);
            }
            return deserialize(nodes);
        }

        // 通过 nodes 列表构造二叉树 -- 后序遍历
        TreeNode deserialize(LinkedList<String> nodes) {
            if (nodes.isEmpty()) return null;

            // 后序遍历得到的 nodes 列表中，最后一个元素是 root 节点的值
            // 从后往前取出元素
            String last = nodes.removeLast();
            if (last.equals(NULL)) return null;
            TreeNode root = new TreeNode(Integer.parseInt(last));

            // 先构造右子树，后构造左子树
            root.right = deserialize(nodes);
            root.left = deserialize(nodes);
            return root;
        }
    }

    /**
     * 剑指 Offer 37. 序列化二叉树 -- 层序遍历
     */
    public static class Codec3 {

        String SEP = ",";
        String NULL = "#";


        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) return "";
            StringBuilder sb = new StringBuilder();

            // 层序遍历
            // 初始化队列，将 root 加入队列
            Queue<TreeNode> q = new LinkedList<>();
            q.offer(root);
            while (!q.isEmpty()) {
                TreeNode cur = q.poll();

                // ---- 层级遍历代码位置 start -----
                if (cur == null) {
                    sb.append(NULL).append(SEP);
                    continue;
                }
                sb.append(cur.val).append(SEP);
                // ---- 层级遍历代码位置 end -----

                q.offer(cur.left);
                q.offer(cur.right);
            }

            return sb.toString();
        }


        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data.isEmpty()) return null;

            // 用队列进行层级遍历，同时用索引 i 记录对应子节点的位置

            String[] nodes = data.split(SEP);
            // 第一个元素就是 root 的值
            TreeNode root = new TreeNode(Integer.parseInt(nodes[0]));

            // 队列 q 记录父节点，将 root 加入队列
            Queue<TreeNode> q = new LinkedList<>();
            q.offer(root);

            for (int i = 1; i < nodes.length; ) {
                // 队列中存的都是父节点
                TreeNode parent = q.poll();
                // 父节点对应的左侧子节点的值
                String left = nodes[i++];
                if (!left.equals(NULL)) {
                    parent.left = new TreeNode(Integer.parseInt(left));
                    q.offer(parent.left);
                } else {
                    parent.left = null;
                }
                // 父节点对应的右侧子节点的值
                String right = nodes[i++];
                if (!right.equals(NULL)) {
                    parent.right = new TreeNode(Integer.parseInt(right));
                    q.offer(parent.right);
                } else {
                    parent.right = null;
                }
            }

            return root;
        }
    }


    /**
     * 剑指 Offer 41. 数据流中的中位数
     */
    static class MedianFinder {

        private PriorityQueue<Integer> large; // 梯形 -- 小堆顶（存放较大的数字）
        private PriorityQueue<Integer> small; // 倒三角形 -- 大堆顶（存放较小的数字）

        /**
         * initialize your data structure here.
         */
        public MedianFinder() {
            large = new PriorityQueue<>();
            small = new PriorityQueue<>((a, b) -> b - a);
        }

        // 不仅要维护large和small的元素个数之差不超过 1，
        // 还要维护large堆的堆顶元素要大于等于small堆的堆顶元素。
        public void addNum(int num) {
            if (small.size() >= large.size()) {
                // 想要往large里添加元素，不能直接添加，而是要先往small里添加，然后再把small的堆顶元素加到large中；
                small.offer(num);
                large.offer(small.poll());
            } else {
                // 向small中添加元素同理
                large.offer(num);
                small.offer(large.poll());
            }
        }

        public double findMedian() {
            if (large.isEmpty() && small.isEmpty()) return -1;
            // 如果元素不一样多，多的那个堆的堆顶元素就是中位数
            if (large.size() < small.size()) {
                return small.peek();
            } else if (large.size() > small.size()) {
                return large.peek();
            }
            return (large.peek() + small.peek()) / 2.0;
        }
    }

    /**
     * 剑指 Offer 43. 1～n 整数中 1 出现的次数
     * <p>
     * 时间复杂度O(logn) 空间复杂度 O(1)
     */
    public int countDigitOne(int n) {
        // 当前位cur, 低位low, 高位 high, 位因子 记为 digit
        int digit = 1, res = 0;
        int high = n / 10, cur = n % 10, low = 0;
        while (high != 0 || cur != 0) {
            if (cur == 0) res += high * digit;
            else if (cur == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;

            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }

    /**
     * 求任意数字出现次数：1～n 整数中 k 出现的次数
     */
    public int digitCounts(int n, int k) {
        long digit = 1;
        int res = 0, high = n / 10, low = 0, cur = n % 10;
        while (high != 0 || cur != 0) {
            if (cur < k) res += high * digit;
            else if (cur == k) res += high * digit + low + 1;
            else res += (high + k) * digit;

            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }

    /**
     * 剑指 Offer 51. 数组中的逆序对
     * <p>
     * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
     * 输入一个数组，求出这个数组中的逆序对的总数。
     * <p>
     * 时间复杂度 O(NlogN) 空间复杂度 O(N)
     */
    public int reversePairs(int[] nums) {
//        int len = nums.length;
//        int res = 0;
//        for (int i = 0; i < len - 1; i++) {
//            for (int j = i + 1; j < len; j++) {
//                if (nums[i] > nums[j]) {
//                    res++;
//                }
//            }
//        }
//        return res;

        this.nums = nums;
        temp = new int[nums.length];
        return mergeSort(0, nums.length - 1);
    }

    int[] nums, temp;

    // 归并排序与逆序对统计
    private int mergeSort(int l, int r) {
        // 1. 终止条件
        if (l >= r) return 0;
        // 2. 递归划分
        int m = (l + r) / 2;
        int res = mergeSort(l, m) + mergeSort(m + 1, r);
        // 3. 合并阶段
        int i = l, j = m + 1;
        for (int k = l; k <= r; k++) {
            // 3.1 暂存数组 nums 闭区间 [i, r] 内的元素至辅助数组 temp
            temp[k] = nums[k];
        }
        // 3.2 循环合并
        for (int k = l; k <= r; k++) {
            if (i == m + 1) {
                // 3.2.1 左子数组已合并完，因此添加右子数组当前元素 temp[j]，并执行 j=j+1;
                nums[k] = temp[j++];
            } else if (j == r + 1) {
                // 3.2.2 右子数组已合并完，因此添加左子数组当前元素 temp[i]，并执行 i=i+1;
                nums[k] = temp[i++];
            } else if (temp[i] <= temp[j]) {
                // 3.2.3 添加左子数组当前元素 temp[i], 并执行 i=i+1;
                nums[k] = temp[i++];
            } else {
                // 3.2.4 即 temp[j] > temp[i]：添加右子数组当前元素 temp[j], 并执行 j=j+1;
                // 此时构成 m - i + 1 个「逆序对」，统计添加至 res
                nums[k] = temp[j++];
                res += m - i + 1; // 统计逆序对
            }
        }
        return res;
    }

    /**
     * 剑指 Offer 64. 求1+2+…+n
     */
    public int sumNum(int n) {
        // if(A && B)  若 A 为 false，则 B 的判断不会执行（即短路），直接判定 A && B 为 false
        // if(A || B)  若 A 为 true ，则 B 的判断不会执行（即短路），直接判定 A || B 为 true

        // 要实现 “当 n = 1 时终止递归”
        // n > 1 && sumNums(n - 1) // 当 n = 1 时 n > 1 不成立 ，此时 “短路” ，终止后续递归
        boolean flag = n > 1 && sumNum(n - 1) > 1;
        sumNumAns += n;
        return sumNumAns;
    }

    int sumNumAns = 0;

    /**
     * 剑指 Offer 59 - I. 滑动窗口的最大值
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        MonotonicQueue window = new MonotonicQueue();
        List<Integer> res = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            if (i < k - 1) {
                window.push(nums[i]);
            } else {
                window.push(nums[i]);
                res.add(window.max());
                window.pop(nums[i - k + 1]);
            }
        }

        int len = res.size();
        int[] ans = new int[len];
        for (int i = 0; i < len; i++) {
            ans[i] = res.get(i);
        }
        return ans;
    }

    // 单调队列
    static class MonotonicQueue {

        LinkedList<Integer> q = new LinkedList<>();

        void push(int num) {
            while (!q.isEmpty() && q.getLast() < num) {
                q.pollLast();
            }
            q.addLast(num);
        }

        void pop(int num) {
            if (num == q.getFirst()) {
                q.pollFirst();
            }
        }

        int max() {
            return q.getFirst();
        }

    }

    /**
     * 4. 寻找两个正序数组的中位数
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if (m == 0 && n != 0) return nums2[n / 2];
        if (m != 0 && n == 0) return nums1[m / 2];
        return 0;
    }

    /**
     * 给定一个有序数组 arr 和 目标值 sum，返回累加和为 sum 的所有不同二元数组
     * <p>
     * 有序数组时间复杂度是 O(N)
     * 无序数组时间复杂度是 O(NlogN)
     * 其中 while 循环时间复杂度是 O(N)，而排序的时间复杂度是 O(NlogN) -- 针对无序数组要先排序
     */
    public static List<List<Integer>> findTwoArray(int[] arr, int sum) {
        List<List<Integer>> ans = new ArrayList<>();
        if (arr == null || arr.length < 2) return ans;

        // 若无序数组先排序
        // sort(arr)

        // 用双指针技巧
        int left = 0, right = arr.length - 1;
        while (left < right) {
            int curSum = arr[left] + arr[right];
            // 记录索引 left 和 right 最初对应的值
            int leftNum = arr[left], rightNum = arr[right];
            if (curSum == sum) {
                // 找到目标和
                List<Integer> curAns = new ArrayList<>();
                curAns.add(arr[left]);
                curAns.add(arr[right]);
                ans.add(curAns);
                // 跳过所有重复的元素
                while (left < right && arr[left] == leftNum) left++;
                while (left < right && arr[right] == rightNum) right--;
            } else if (curSum < sum) {
                //left++;
                while (left < right && arr[left] == leftNum) left++; // 跳过重复元素
            } else if (curSum > sum) {
                //right--;
                while (left < right && arr[right] == rightNum) right--; // 跳过重复元素
            }
        }
        return ans;
    }

    /**
     * 给定一个有序数组 arr 和 目标值 sum，返回累加和为 sum 的所有不同三元数组
     */
    public static List<List<Integer>> findThreeArray(int[] arr, int sum) {
        List<List<Integer>> ans = new ArrayList<>();
        if (arr == null || arr.length < 3) return ans;

        for (int i = arr.length - 1; i >= 0; i--) {
            // 每次把最后一个数提取除来
            int max = arr[i];
            int target = sum - max;
            if (i != arr.length - 1 && arr[i] == arr[i + 1]) continue; // 去重
            // 在剩下的数组中寻找所有不同的二元数组
            int[] leftArr = Arrays.copyOfRange(arr, 0, i);
            List<List<Integer>> twoArray = findTwoArray(leftArr, target);
            if (twoArray.size() > 0) {
                for (List<Integer> list : twoArray) {
                    list.add(max);
//                        if (!ans.contains(list)){
//                            ans.add(list);
//                        }
                }
                ans.addAll(twoArray);
            }
        }
        return ans;
    }

    /**
     * 给定一个有序数组 arr 和 目标值 sum，返回累加和为 sum 的所有不同三元数组
     * <p>
     * 时间复杂度：O(N^2)
     */
    public static List<List<Integer>> threeSumTarget(int[] arr, int sum) {
        List<List<Integer>> ans = new ArrayList<>();
        if (arr == null || arr.length < 3) return ans;

        int n = arr.length;
        for (int i = 0; i < n; i++) {
            // 对 sum - arr[i] 计算 twoSum
            List<List<Integer>> twoArray = twoSumTarget(arr, i + 1, sum - arr[i]);
            if (twoArray.size() > 0) {
                for (List<Integer> list : twoArray) {
                    list.add(arr[i]);
                }
                ans.addAll(twoArray);
            }
            // 跳过第一个数字重复的情况，否则会出现重复结果
            while (i < n - 1 && arr[i] == arr[i + 1]) i++;
        }
        return ans;
    }

    /**
     * 从 arr[start] 开始，计算有序数组 arr 中所有和为 target 的二元组
     */
    public static List<List<Integer>> twoSumTarget(int[] arr, int start, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        int left = start, right = arr.length - 1;
        while (left < right) {
            int curSum = arr[left] + arr[right];
            // 记录索引 left 和 right 最初对应的值
            int leftNum = arr[left], rightNum = arr[right];
            if (curSum == target) {
                // 找到目标和
                List<Integer> curAns = new ArrayList<>();
                curAns.add(arr[left]);
                curAns.add(arr[right]);
                ans.add(curAns);
                // 跳过所有重复的元素
                while (left < right && arr[left] == leftNum) left++;
                while (left < right && arr[right] == rightNum) right--;
            } else if (curSum < target) {
                //left++;
                while (left < right && arr[left] == leftNum) left++; // 跳过重复元素
            } else if (curSum > target) {
                //right--;
                while (left < right && arr[right] == rightNum) right--; // 跳过重复元素
            }
        }

        return ans;
    }

    /**
     * 给定一个有序数组 arr 和 目标值 sum，返回累加和为 sum 的所有不同四元数组
     * <p>
     * 时间复杂度：O(N^3)
     */
    public static List<List<Integer>> fourSumTarget(int[] arr, int sum) {
        List<List<Integer>> ans = new ArrayList<>();
        if (arr == null || arr.length < 4) return ans;

        return nSumTarget(arr, 4, 0, sum);
    }

    /**
     * 给定一个有序数组 arr 和 目标值 target，返回累加和为 target 的所有不同 n 元数组
     * <p>
     * 时间复杂度：O(N^n)
     */
    public static List<List<Integer>> nSumTarget(int[] arr, int n, int start, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        if (arr == null || arr.length < n || n < 2) return ans;

        Arrays.sort(arr);

        int size = arr.length;
        // base case
        if (n == 2) {
            // 双指针操作
            int left = start, right = size - 1;
            while (left < right) {
                int sum = arr[left] + arr[right];
                int leftNum = arr[left], rightNum = arr[right];
                if (sum == target) {
                    List<Integer> list = new ArrayList<>();
                    list.add(arr[left]);
                    list.add(arr[right]);
                    ans.add(list);
                    // 跳过所有重复的元素
                    while (left < right && arr[left] == leftNum) left++;
                    while (left < right && arr[right] == rightNum) right--;
                } else if (sum < target) {
                    while (left < right && arr[left] == leftNum) left++;
                } else if (sum > target) {
                    while (left < right && arr[right] == rightNum) right--;
                }
            }
        } else {
            // n > 2 时，递归计算 (n-1)Sum 的结果
            for (int i = start; i < size; i++) {
                List<List<Integer>> list = nSumTarget(arr, n - 1, i + 1, target - arr[i]);
                for (List<Integer> list1 : list) {
                    // (n-1)Sum 加上 nums[i] 就是 nSum
                    list1.add(arr[i]);
                    ans.add(list1);
                }
                while (i < size - 1 && arr[i] == arr[i + 1]) i++;
            }
        }
        return ans;
    }

    /**
     * 给定一个二维数组 matrix，其中每个数都是正数，从左到右下每一步只能向右或向下，沿途数字累加，返回最小路径和
     */
    public static int minSum(int[][] matrix) {
        if (matrix == null || matrix.length == 0) return 0;

        return process(matrix, matrix.length - 1, matrix[0].length - 1);
    }

    static HashMap<String, Integer> minMemo = new HashMap<>();

    // 定义： [0,0] 到 [i,j] 的最小路径和
    public static int process(int[][] matrix, int i, int j) {

        if (i < 0 || j < 0) return Integer.MAX_VALUE;

        if (i == 0 && j == 0) return matrix[0][0];

        String key = i + "," + j;
        if (minMemo.containsKey(key)) return minMemo.get(key);

        minMemo.put(key, matrix[i][j] + Math.min(process(matrix, i - 1, j), process(matrix, i, j - 1)));

        return minMemo.get(key);
    }


    /**
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
     * 每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
     * <p>
     * 输入: 2
     * 输出: 1
     * 解释: 2 = 1 + 1, 1 × 1 = 1
     * <p>
     * 输入: 10
     * 输出: 36
     * 解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
     */
    public int cuttingRope(int n) {
        if (n == 2) return 1;
        if (n == 3) return 2;
        if (n == 4) return 4;

        int len = n;
        int res = 1;
        // 3是最小切分段的乘积最大
        while (len > 4) {
            res *= 3;
            len -= 3;
        }
        return res * len;
    }

    public int cuttingRope2(int n) {
        // dp 数组：dp[i]表示长度为i的绳子剪成m端后长度的最大乘积(m>1)
        int[] dp = new int[n + 1];

        dp[2] = 1;
        for (int i = 3; i <= n; i++) {
            // 首先对绳子剪长度为j的一段,其中取值范围为 2 <= j < i
            for (int j = 2; j < i; j++) {
                // Math.max(j*(i-j),j*dp[i-j]是由于减去第一段长度为j的绳子后，可以继续剪也可以不剪
                dp[i] = Math.max(dp[i], Math.max(j * (i - j), j * dp[i - j]));
            }
        }

        return dp[n];
    }

    /**
     * 去除重复字母
     * <p>
     * 给一个仅包含小写字母的字符串，去除重复字母，使得每个字母只出现移除。需保证返回结果的字典序最小
     * 要求不能打乱其他字符的相对位置
     * <p>
     * 如："bcabc" --> "abc", "cbacdcbc" --> "acdb"
     * <p>
     * 要求一、要去重。
     * <p>
     * 要求二、去重字符串中的字符顺序不能打乱s中字符出现的相对顺序。
     * <p>
     * 要求三、在所有符合上一条要求的去重字符串中，字典序最小的作为最终结果。
     */
    private String removeDuplicateLetters(String s) {
        // 存放去重结果
        Stack<Character> stk = new Stack<>();

        // 计数器用于记录字符串中字符的数量
        int[] count = new int[256];
        for (int i = 0; i < s.length(); i++) {
            count[s.charAt(i)]++;
        }

        // 记录栈中是否存在某个字符
        boolean[] inStack = new boolean[256];

        for (char c : s.toCharArray()) {
            // 每遍历过一个字符，都将对应的计数减一
            count[c]--;

            if (inStack[c]) continue;

            // 插入之前，和之前的元素比较一下大小，如果字典序比前面的小，pop 前面的元素
            while (!stk.isEmpty() && stk.peek() > c) {
                if (count[stk.peek()] == 0) {
                    // 若之后不存在栈顶元素了，则停止 pop
                    break;
                }
                inStack[stk.pop()] = false;// 弹出栈顶元素，并把该元素标记为不在栈中
            }

            stk.push(c);
            inStack[c] = true;
        }

        StringBuilder sb = new StringBuilder();
        while (!stk.isEmpty()) {
            sb.append(stk.pop());
        }

        return sb.reverse().toString();
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
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
