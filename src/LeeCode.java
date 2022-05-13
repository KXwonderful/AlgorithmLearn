import java.util.*;

public class LeeCode {

    /**
     * 剑指 Offer 46. 把数字翻译成字符串
     * <p>
     * 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
     * 一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
     * <p>
     * 输入: 12258
     * 输出: 5
     * 解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
     */
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int len = s.length();

        // dp[i] 代表以 i 为结尾的数字的翻译方案数量。
        // 区间 [10, 25]：dp[i] = dp[i-1] + dp[i-2]
        // 区间 [0, 10) V (25, 99]：dp[i] = dp[i-1]
        int[] dp = new int[len + 1];
        // base case
        dp[0] = dp[1] = 1; // 即 “无数字” 和 “第 1 位数字” 的翻译方法数量均为 1
        for (int i = 2; i <= len; i++) {
            String temp = s.substring(i - 2, i);
            dp[i] = temp.compareTo("10") >= 0 && temp.compareTo("25") <= 0 ? dp[i - 1] + dp[i - 2] : dp[i - 1];
        }

        return dp[len];
    }

    /**
     * 剑指 Offer 47. 礼物的最大价值
     * <p>
     * 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，
     * 并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
     * <p>
     * 输入:
     * [
     * [1,3,1],
     * [1,5,1],
     * [4,2,1]
     * ]
     * 输出: 12
     * 解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
     */
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                memo[i][j] = 1;
            }
        }

        return dp(grid, m - 1, n - 1);
    }

    int[][] memo;

    // dp:（0，0）到（i，j）的最大路径和
    int dp(int[][] grid, int i, int j) {
        if (i < 0 || j < 0) return Integer.MIN_VALUE;

        // base case
        if (i == 0 && j == 0) return grid[i][j];

        if (memo[i][j] != -1) return memo[i][j];

        memo[i][j] = grid[i][j] + Math.max(dp(grid, i - 1, j), dp(grid, i, j - 1));

        return memo[i][j];
    }

    /**
     * 剑指 Offer 48. 最长不含重复字符的子字符串
     * <p>
     * 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
     * <p>
     * 输入: "abcabcbb"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
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
     * 剑指 Offer 49. 丑数
     * <p>
     * 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
     * <p>
     * 输入: n = 10
     * 输出: 12
     * 解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
     */
    public int nthUglyNumber(int n) {
        // dp[i]: 第 i+1 个丑数
        int[] dp = new int[n];

        // base case
        dp[0] = 1;

        // 根据丑数性质：
        // dp[i] 是 dp[i-1] 最近的丑数，索引a，b，c满足以下条件：
        // dp[a]*2 > dp[i] >= dp[a-1]*2
        // dp[b]*3 > dp[i] >= dp[b-1]*3
        // dp[c]*5 > dp[i] >= dp[c-1]*5
        int a = 0, b = 0, c = 0;
        for (int i = 1; i < n; i++) {
            int n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5;
            dp[i] = Math.min(Math.min(n2, n3), n5);
            if (dp[i] == n2) a++;
            if (dp[i] == n3) b++;
            if (dp[i] == n5) c++;
        }

        return dp[n - 1];
    }


    /**
     * 剑指 Offer 50. 第一个只出现一次的字符
     * <p>
     * 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
     * <p>
     * 输入：s = "abaccdeff"
     * 输出：'b'
     */
    public char firstUniqChar(String s) {
        HashMap<Character, Integer> memo = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            memo.put(c, memo.getOrDefault(c, 0) + 1);
        }

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (memo.get(c) == 1) {
                return c;
            }
        }
        return ' ';
    }

    /**
     * 剑指 Offer 51. 数组中的逆序对
     * <p>
     * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
     * <p>
     * 输入: [7,5,6,4]
     * 输出: 5
     */
    public int reversePairs(int[] nums) {
        return 0;
    }

    /**
     * 剑指 Offer 52. 两个链表的第一个公共节点
     * <p>
     * 输入两个链表，找出它们的第一个公共节点。
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        while (p1 != p2) {
            if (p1 == null) {
                p1 = headB;
            } else {
                p1 = p1.next;
            }

            if (p2 == null) {
                p2 = headA;
            } else {
                p2 = p2.next;
            }
        }

        return p1;
    }

    /**
     * 剑指 Offer 53 - I. 在排序数组中查找数字 I
     * <p>
     * 统计一个数字在排序数组中出现的次数。
     */
    public int search(int[] nums, int target) {
        int left = left_bound(nums, target);
        if (left == -1) return 0;
        int right = right_bound(nums, target);
        return right - left + 1;
    }

    // 左边界二分搜索
    int left_bound(int[] nums, int target) {
        int len = nums.length;
        int left = 0, right = len - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                // 收缩右边界
                right = mid - 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (left >= len || nums[left] != target) return -1;
        return left;
    }

    // 右边界二分搜索
    int right_bound(int[] nums, int target) {
        int len = nums.length;
        int left = 0, right = len - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                // 收缩左边界
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (right < 0 || nums[right] != target) return -1;
        return right;
    }


    /**
     * 剑指 Offer 53 - II. 0～n-1中缺失的数字
     * <p>
     * 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。
     * 在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
     * <p>
     * 输入: [0,1,3]
     * 输出: 2
     */
    public int missingNumber(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == mid) {
                left++;
            } else if (nums[mid] > mid) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    /**
     * 剑指 Offer 54. 二叉搜索树的第k大节点
     * <p>
     * 给定一棵二叉搜索树，请找出其中第 k 大的节点的值。
     */
    public int kthLargest(TreeNode root, int k) {
        traverse(root);
        int len = nums.size();
        if (k <= len) return nums.get(len - k);
        return -1;
    }

    List<Integer> nums = new ArrayList<>();

    void traverse(TreeNode root) {
        // 中序遍历有序
        if (root == null) return;

        traverse(root.left);
        nums.add(root.val);
        traverse(root.right);
    }


    /**
     * 剑指 Offer 55 - I. 二叉树的深度
     * <p>
     * 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
     */
    public int maxDepth1(TreeNode root) {
        if (root == null) return 0;

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int res = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            res++;
            for (int i = 0; i < size; i++) {
                TreeNode temp = q.poll();
                if (temp.left != null) q.offer(temp.left);
                if (temp.right != null) q.offer(temp.right);
            }
        }
        return res;
    }

    // 回溯
    public int maxDepth2(TreeNode root) {
        backtrack(root);
        return res;
    }

    int depth = 0;
    int res = 0;

    void backtrack(TreeNode root) {
        if (root == null) return;

        depth++;
        // 遍历的过程中记录最大深度
        res = Math.max(res, depth);
        backtrack(root.left);
        backtrack(root.right);
        depth--;
    }

    // 动态规划: 定义：输入一个节点，返回以该节点为根的二叉树的最大深度
    public int maxDepth3(TreeNode root) {
        if (root == null) return 0;
        int leftMax = maxDepth3(root.left);
        int rightMax = maxDepth3(root.right);
        // 根据左右子树的最大深度推出原二叉树的最大深度
        return 1 + Math.max(leftMax, rightMax);
    }


    /**
     * 剑指 Offer 55 - II. 平衡二叉树
     * <p>
     * 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
     */
    public boolean isBalanced(TreeNode root) {
        maxDepth(root);
        return isBalance;
    }

    boolean isBalance = true;

    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        if (Math.abs(leftDepth - rightDepth) > 1) {
            isBalance = false;
        }
        return 1 + Math.max(leftDepth, rightDepth);
    }


    /**
     * 剑指 Offer 56 - I. 数组中数字出现的次数
     * <p>
     * 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。
     * 请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
     */
    public int[] singleNumbers(int[] nums) {
        int x = 0, y = 0, n = 0, m = 1;
        // 1. 遍历异或
        for (int num : nums) {
            n ^= num;
        }
        // 2. 循环左移，计算m
        while ((n & m) == 0) {
            m <<= 1;
        }
        // 3. 遍历 nums 分组
        for (int num : nums) {
            if ((num & m) != 0) {
                // 此时问题已经退化为数组中有一个数字只出现了一次
                x ^= num; // 3.1 当 num & m != 0
            } else {
                y ^= num; // 3.2 当 num & m == 0
            }
        }

        return new int[]{x, y};
    }

    /**
     * 一个整型数组 nums 里除1个数字之外，其他数字都出现了两次。
     * <p>
     * 异或运算: 两个相同数字异或为 0 ，即对于任意整数 a 有 a ^ a = 0
     * 异或运算: 满足交换律 a^b = b^a，即以上运算结果与 nums的元素顺序无关
     */
    public int singleNumber(int[] nums) {
        int x = 0;
        for (int num : nums) {
            x ^= num;
        }
        return x;
    }

    /**
     * 剑指 Offer 56 - II. 数组中数字出现的次数 II
     * <p>
     * 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
     * <p>
     * 异或运算：x ^ 0 = x ， x ^ 1 = ~x
     * 与运算：x & 0 = 0 ， x & 1 = x
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
     * 剑指 Offer 57. 和为s的两个数字
     * <p>
     * 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
     */
    public int[] twoSum(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int[] res = new int[2];
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum == target) {
                res[0] = nums[left];
                res[1] = nums[right];
                return res;
            } else if (sum > target) {
                right--;
            } else {
                left++;
            }
        }
        return res;
    }

    /**
     * 剑指 Offer 57 - II. 和为s的连续正数序列
     * <p>
     * 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
     * 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
     */
    public int[][] findContinuousSequence(int target) {
        List<int[]> res = new ArrayList<>();

        int left = 1, right = 1;
        int sum = 0; // 看作是 window
        while (right <= target / 2) {
            if (sum < target) {
                // 右移窗口
                sum += right;
                right++;
            } else if (sum > target) {
                // 收缩窗口
                sum -= left;
                left++;
            } else {
                int[] arr = new int[right - left];
                for (int i = left; i < right; i++) {
                    arr[i - left] = i;
                }
                res.add(arr);
                // 收缩窗口后继续
                sum -= left;
                left++;
            }
        }

        return res.toArray(new int[res.size()][]);
    }

    /**
     * 剑指 Offer 58 - I. 翻转单词顺序
     * <p>
     * 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。
     * 例如输入字符串"I am a student. "，则输出"student. a am I"。
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
     * 剑指 Offer 58 - II. 左旋转字符串
     * <p>
     * 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
     * 比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
     */
    public String reverseLeftWords(String s, int n) {
        String temp = s.substring(0, n);
        return s.substring(n) + temp;
    }

    /**
     * 剑指 Offer 59 - I. 滑动窗口的最大值
     * <p>
     * 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0) return new int[0];

        MonotonousQueue q = new MonotonousQueue();
        int[] res = new int[nums.length - k + 1];
        for (int i = 0; i < nums.length; i++) {
            if (i < k - 1) {
                q.push(nums[i]);
            } else {
                q.push(nums[i]);
                res[i - k + 1] = q.max();
                q.pop(nums[i - k + 1]);
            }
        }
        return res;
    }

    // 定义单调队列
    public class MonotonousQueue {
        LinkedList<Integer> q = new LinkedList<>();

        // 队尾添加元素
        void push(int num) {
            // 把比num小的队尾元素移除，确保单调队列
            while (!q.isEmpty() && q.getLast() < num) {
                q.pollLast();
            }
            q.addLast(num);
        }

        // 若 num 和对头元素相等则移除元素
        void pop(int num) {
            if (num == q.getFirst()) {
                q.pollFirst();
            }
        }

        // 最大元素
        int max() {
            return q.getFirst();
        }
    }

    /**
     * 剑指 Offer 59 - II. 队列的最大值
     * <p>
     * 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。
     * 若队列为空，pop_front 和 max_value 需要返回 -1
     */
    class MaxQueue {

        Queue<Integer> q1;
        Deque<Integer> q2;

        public MaxQueue() {
            q1 = new LinkedList<>(); // 数据队
            q2 = new LinkedList<>(); // 双向队列，队首存最大元素
        }

        public int max_value() {
            return q2.isEmpty() ? -1 : q2.getFirst();
        }

        public void push_back(int value) {
            q1.offer(value);
            // 将双向队列中队尾所有小于 value 的元素弹出（以保持 deque 非单调递减），并将元素 value 入队 deque
            while (!q2.isEmpty() && q2.getLast() < value) {
                q2.pollLast();
            }
            q2.offerLast(value);
        }

        public int pop_front() {
            if (q1.isEmpty()) return -1;
            int pop = q1.poll();
            if (!q2.isEmpty() && pop == q2.getFirst()) {
                q2.pollFirst();
            }
            return pop;
        }
    }

    /**
     * 剑指 Offer 60. n个骰子的点数
     * <p>
     * 把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
     * 你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。
     */
    public double[] dicesProbability(int n) {
        double[] dp = new double[6];
        Arrays.fill(dp, 1.0 / 6.0);

        for (int i = 2; i <= n; i++) {
            // i个骰子点数之和的个数为 6*i-(i-1)，化简：5*i+1
            double[] temp = new double[5 * i + 1];

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
     * 剑指 Offer 61. 扑克牌中的顺子
     * <p>
     * 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。
     * 2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
     */
    public boolean isStraight(int[] nums) {
        Set<Integer> repeat = new HashSet<>();
        int max = Integer.MIN_VALUE, min = Integer.MAX_VALUE;
        for (int num : nums) {
            if (num == 0) continue;
            if (repeat.contains(num)) return false;
            max = Math.max(max, num);
            min = Math.min(min, num);
            repeat.add(num);
        }
        return max - min < 5;
    }

    /**
     * 剑指 Offer 62. 圆圈中最后剩下的数字
     * <p>
     * 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。
     * 例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
     */
    public int lastRemaining(int n, int m) {
        int res = 0;
        for (int i = 2; i <= n; i++) {
            res = (res + m) % i;
        }
        return res;
    }

    /**
     * 剑指 Offer 63. 股票的最大利润
     * <p>
     * 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
     */
    public int maxProfit(int[] prices) {
        if (prices.length < 1) return 0;
        int curMin = prices[0];
        int res = 0;
        for (int sell = 1; sell < prices.length; sell++) {
            curMin = Math.min(curMin, prices[sell]);
            res = Math.max(res, prices[sell] - curMin);
        }
        return res;
    }

    /**
     * 剑指 Offer 64. 求1+2+…+n
     * <p>
     * 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
     */
    public int sumNums(int n) {
        // if(A && B)  若 A 为 false，则 B 的判断不会执行（即短路），直接判定 A && B 为 false
        // if(A || B)  若 A 为 true ，则 B 的判断不会执行（即短路），直接判定 A || B 为 true

        // 要实现 “当 n = 1 时终止递归”
        // n > 1 && sumNums(n - 1) // 当 n = 1 时 n > 1 不成立 ，此时 “短路” ，终止后续递归
        boolean flag = n > 1 && sumNums(n - 1) > 1;
        sumNums += n;
        return sumNums;
    }

    int sumNums = 0;

    /**
     * 剑指 Offer 65. 不用加减乘除做加法
     * <p>
     * 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
     * <p>
     * a & b 的结果往左移1位就是a+b进位的结果
     * a ^ b 的结果是a+b无进位的结果
     */
    public int add(int a, int b) {
        // 递归
        // if (b == 0) return a;
        // return add(a ^ b, (a & b) << 1);

        int sum = a;
        // 直到进位信息消失为0
        while (b != 0) {
            sum = a ^ b;     // 计算无进位相加的信息 -> sum
            b = (a & b) << 1;// 计算进位信息 -> b
            a = sum;         // 把无进位相加的信息赋值给 a
        }
        return sum;
    }

    /**
     * 剑指 Offer 66. 构建乘积数组
     * <p>
     * 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中B[i] 的值是数组 A 中除了下标 i 以外的元素的积,
     * 即B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法
     */
    public int[] constructArr(int[] a) {
        int len = a.length;
        if (len == 0) return new int[0];

        int[] b = new int[len];
        b[0] = 1;
        int temp = 1;

        // 左下三角
        for (int i = 1; i < len; i++) {
            b[i] = b[i - 1] * a[i - 1];
        }
        // 右上三角
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
        char[] chars = str.trim().toCharArray();
        boolean isRemoveStartZero = false;
        boolean isAppearZero = false;
        boolean isPositiveInteger = true;

        StringBuilder sb = new StringBuilder();
        for (char c : chars) {
            if (isRemoveStartZero) {
                if (c == ' ' || c < '0' || c > '9') {
                    break; // 出现空格或非数字停止
                }
                sb.append(c);
            }
            if (!isRemoveStartZero && c != '0') {
                isRemoveStartZero = true;
                if (c == '+') {
                    if (isAppearZero) return 0;
                    isPositiveInteger = true;
                } else if (c == '-') {
                    if (isAppearZero) return 0;
                    isPositiveInteger = false;
                } else if (c == ' ' || c < '1' || c > '9') {
                    return 0;
                } else {
                    sb.append(c);
                }
            }
            if (!isRemoveStartZero && c == '0') {
                isAppearZero = true;
            }
        }

        if (sb.length() == 0) return 0;

        double res = 0;
        String s = sb.toString();
        for (int i = 0; i < s.length(); i++) {
            res = res * 10 + (s.charAt(i) - '0');
        }

        if (isPositiveInteger) {
            if (res > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        } else {
            if (-res < Integer.MIN_VALUE) return Integer.MIN_VALUE;
            res = -res;
        }

        return (int) res;
    }

    /**
     * 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
     * <p>
     * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //1、如果 p 和 q 都比当前节点小，那么显然 p 和 q 都在左子树，那么 在左子树。
        //2、如果 p 和 q 都比当前节点大，那么显然 p 和 q 都在右子树，那么 在右子树。
        //3、一旦发现 p 和 q 在当前节点的两侧，说明当前节点就是 LCA。
        if (root == null) return null;

        if (p.val > q.val) return lowestCommonAncestor(root, q, p);

        if (root.val >= p.val && root.val <= q.val) {
            //  p <= root <= q
            return root;
        }

        if (q.val <= root.val) {
            // 在左子树
            return lowestCommonAncestor(root.left, p, q);
        } else {
            // 在右子树
            return lowestCommonAncestor(root.right, p, q);
        }
    }

    /**
     * 剑指 Offer 68 - II. 二叉树的最近公共祖先
     * <p>
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     */
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        // 情况 1，如果p和q都在以root为根的树中，函数返回的即使p和q的最近公共祖先节点。
        // 情况 2，那如果p和q都不在以root为根的树中怎么办呢？函数理所当然地返回null呗。
        // 情况 3，那如果p和q只有一个存在于root为根的树中呢？函数就会返回那个节点。

        if (root == null) return null;
        if (root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);

        // 1.
        if (left != null && right != null) return root;
        // 2.
        if (left == null && right == null) return null;
        // 3.
        return left == null ? right : left;
    }

    /**
     * 面试题19. 正则表达式匹配
     * <p>
     * 请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。
     * 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配
     * <p>
     * s 和 p 相互匹配的过程大致是，两个指针 i, j 分别在 s 和 p 上移动，如果最后两个指针都能移动到字符串的末尾，那么就匹配成功，反之则匹配失败。
     * 正则表达算法问题只需要把住一个基本点：看 s[i] 和 p[j] 两个字符是否匹配，一切逻辑围绕匹配/不匹配两种情况展开即可。
     */
    public boolean isMatch(String s, String p) {
        matchMemo = new HashMap<>();
        return dp(s, 0, p, 0);
    }

    HashMap<String, Boolean> matchMemo;

    // dp 定义：s[i...] 可以匹配 p[...]
    boolean dp(String s, int i, String p, int j) {
        // base case
        if (j == p.length()) return i == s.length();
        if (i == s.length()) {
            // 此时 p 若还有，则后面一定是字符和 * 成对出现
            if ((p.length() - j) % 2 == 1) return false;
            // 检查是否为 x*y*z* 这种形式
            for (; j + 1 < p.length(); j += 2) {
                if (p.charAt(j + 1) != '*') return false;
            }
            return true;
        }

        // 备忘录
        String key = i + "," + j;
        if (matchMemo.containsKey(key)) return matchMemo.get(key);
        boolean res;

        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            // 匹配
            if (j < p.length() - 1 && p.charAt(j + 1) == '*') {
                // 1.1 通配符匹配 0 次 or 多次
                res = dp(s, i, p, j + 2) || dp(s, i + 1, p, j);
            } else {
                // 1.2 通配符匹配 1 次
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
        matchMemo.put(key, res);
        return res;
    }

    /**
     * 面试题32 - I. 从上到下打印二叉树
     * <p>
     * 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印
     */
    public int[] levelOrder(TreeNode root) {
        if (root == null) return new int[0];

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        List<Integer> list = new ArrayList<>();

        while (!q.isEmpty()) {
            TreeNode cur = q.poll();
            list.add(cur.val);

            if (cur.left != null) q.offer(cur.left);
            if (cur.right != null) q.offer(cur.right);
        }

        int size = list.size();
        int[] res = new int[size];
        for (int i = 0; i < size; i++) {
            res[i] = list.get(i);
        }

        return res;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }
}
