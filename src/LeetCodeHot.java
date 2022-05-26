import java.util.*;

public class LeetCodeHot {

    /**
     * 31. 下一个排列
     * arr = [1,2,3] 的下一个排列是 [1,3,2]
     * 如何变大：从低位挑一个大一点的数，交换前面一个小一点的数。
     * 变大的幅度要尽量小
     */
    public static void nextPermutation(int[] nums) {
        int len = nums.length;
        //1. 从右往左，寻找第一个比右邻居小的数，把它换到后面去(此时记录下位置，不用真换)
        int firstSmallIndex = -1;
        for (int i = len - 1; i > 0; i--) {
            if (nums[i] > nums[i - 1]) {
                //swap(nums, i, i-1);
                firstSmallIndex = i - 1;
                break;
            }
        }
        if (firstSmallIndex == -1) {
            // 说明已经是最大的排列，反转当前排列即可
            reverse(nums, 0, len - 1);
            return;
        }

        //2. 接着还是从右往左，寻找第一个比这个 2 大的数
        for (int i = len - 1; i > firstSmallIndex; i--) {
            if (nums[i] > nums[firstSmallIndex]) {
                // 交换这两个数
                swap(nums, i, firstSmallIndex);
                break;
            }
        }

        // 3. 此时后面的肯定是递减的， 翻转即可
        reverse(nums, firstSmallIndex + 1, len - 1);

    }

    /**
     * 翻转从下标 start 到 end 的数组 arr
     */
    public static void reverse(int[] arr, int start, int end) {
        while (start < end) {
            swap(arr, start, end);
            start++;
            end--;
        }
    }

    //==================================================================================================================

    /**
     * 32. 最长有效括号
     * 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
     */
    public static int longestValidParentheses(String s) {
        if (s == null || s.length() == 0) return 0;
        Stack<Integer> stack = new Stack<>();
        int length = s.length();
        // dp[i] 的定义：记录以 s[i-1] 结尾的最长合法括号子串长度
        int[] dp = new int[length + 1];
        // ")()())()"
        for (int i = 0; i < length; i++) {
            if (s.charAt(i) == '(') {
                // 遇到左括号，记录索引
                stack.push(i);
                dp[i + 1] = 0;
            } else {
                if (!stack.isEmpty()) {
                    int leftIndex = stack.pop();
                    // 以这个右括号结尾的最长子串长度
                    dp[i + 1] = 1 + i - leftIndex + dp[leftIndex];
                } else {
                    dp[i + 1] = 0;
                }
            }
        }
        int res = 0;
        for (int j : dp) {
            res = Math.max(res, j);
        }
        return res;
    }

    //==================================================================================================================

    /**
     * 33. 搜索旋转排序数组
     * 整数数组 nums 按升序排列，数组中的值 互不相同
     * 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target
     * 如：nums = [4,5,6,7,0,1,2], target = 0
     * 输出 4
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) return -1;
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            // 前半部分有序,注意此处用小于等于
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    // target在前半部分
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (target <= nums[right] && target > nums[mid]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    //==================================================================================================================

    /**
     * 39. 组合总和
     * 给你一个 无重复元素 的整数数组candidates 和一个目标整数target，找出candidates中可以使数字和为目标数target 的所有不同组合 ，
     * 并以列表形式返回。你可以按 任意顺序 返回这些组合。
     * <p>
     * 如：candidates = [2,3,6,7], target = 7
     * 输出：[[2,2,3],[7]]
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates.length == 0) return res;
        backtrack(candidates, 0, target, 0);
        return res;
    }

    List<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> track = new LinkedList<>();

    void backtrack(int[] candidates, int start, int target, int sum) {

        // 找到目标和
        if (sum == target) {
            res.add(new LinkedList<>(track));
            return;
        }
        // 超过目标和，直接结束
        if (sum > target) return;

        // 回溯算法框架
        for (int i = start; i < candidates.length; i++) {
            // 选择
            track.add(candidates[i]);
            sum += candidates[i];
            backtrack(candidates, i, target, sum);
            // 撤销选择
            sum -= candidates[i];
            track.removeLast();
        }
    }

    //==================================================================================================================

    /**
     * 53. 最大子数组和
     * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     */
    public static int maxSubArray(int[] nums) {
        int len = nums.length;
        if (len == 0) return 0;
        // dp[i]：以 nums[i] 为结尾的「最大子数组和」
        int[] dp = new int[len];
        dp[0] = nums[0];
        for (int i = 1; i < len; i++) {
            dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
        }

        int res = Integer.MIN_VALUE;
        for (int num : dp) {
            res = Math.max(res, num);
        }

        return res;
    }

    // 状态压缩
    public static int maxSubArray2(int[] nums) {
        int len = nums.length;
        if (len == 0) return 0;
        int pre = nums[0];
        int res = nums[0];
        for (int i = 1; i < len; i++) {
            int temp = Math.max(nums[i], nums[i] + pre);
            pre = temp;
            res = Math.max(res, temp);
        }
        return res;
    }


    //==================================================================================================================

    /**
     * 70. 爬楼梯
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     */
    public static int climbStairs(int n) {
        cMemo = new int[n + 1];
        return dp(n);
    }

    private static int[] cMemo;

    private static int dp(int n) {
        if (n <= 2) {
            return n;
        }
        if (cMemo[n] > 0) return cMemo[n];

        cMemo[n] = dp(n - 1) + dp(n - 2);

        return cMemo[n];
    }

    //==================================================================================================================

    /**
     * 128. 最长连续序列
     * 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
     * 请你设计并实现时间复杂度为O(n) 的算法解决此问题。
     * <p>
     * 如：nums = [100,4,200,1,3,2]
     * 输出：4
     * 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
     */
    public static int longestConsecutive(int[] nums) {
        // 用哈希
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int res = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int temp = num; // 以当前数向后枚举
                while (set.contains(temp + 1)) {
                    temp++;
                }
                // 更新答案
                res = Math.max(res, temp - num + 1);
            }
        }
        return res;
    }

    //==================================================================================================================

    /**
     * 139. 单词拆分
     * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
     */
    public static boolean wordBreak(String s, List<String> wordDict) {
        memo = new int[s.length()];
        Arrays.fill(memo, -1);

        return dp(s, 0, wordDict);
    }

    private static int[] memo;

    // 定义：返回 s[i..] 是否能够被 wordDict 拼出
    private static boolean dp(String s, int i, List<String> wordDict) {
        // base case
        if (i == s.length()) {
            return true;
        }

        if (memo[i] != -1) return memo[i] == 1;

        for (String word : wordDict) {
            int len = word.length();
            if (i + len > s.length()) continue;

            String subStr = s.substring(i, i + len);

            if (!subStr.equals(word)) continue;

            if (dp(s, i + len, wordDict)) {
                memo[i] = 1;
                return true;
            }
        }
        // s[i..] 不能被拼出，结果记入备忘录
        memo[i] = 0;
        return false;
    }

    //==================================================================================================================

    /**
     * 148. 排序链表
     */
    public static ListNode sortList(ListNode head) {
        if (head == null) return null;

        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> (a.val - b.val));
        ListNode cur = head;
        while (cur != null) {
            pq.add(new ListNode(cur.val));
            cur = cur.next;
        }
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;
        while (!pq.isEmpty()) {
            p.next = pq.poll();
            p = p.next;
        }
        return dummy.next;
    }

    // 归并法
    public static ListNode sortList2(ListNode head) {
        if (head == null) return null;

        // 寻找中间点
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode temp = slow.next;
        slow.next = null; // 把head中间切割成两半
        ListNode left = sortList2(head);
        ListNode right = sortList2(temp);

        return mergeList(left, right);
    }

    // 合并两个有序链表
    private static ListNode mergeList(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;
        ListNode p1 = l1, p2 = l2;
        while (p1 != null && p2 != null) {
            if (p1.val <= p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }

        if (p1 != null) p.next = p1;
        if (p2 != null) p.next = p2;
        return dummy.next;
    }


    //==================================================================================================================

    /**
     * 152. 乘积最大子数组
     * 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
     * 输入: nums = [2,3,-2,4]
     * 输出: 6
     * 解释: 子数组 [2,3] 有最大乘积 6。
     */
    public static int maxProduct(int[] nums) {
        // 参考 53 题
        int len = nums.length;
        if (len == 0) return 0;
        // maxDp[i]：以 num[i] 为结尾的连续子数组最大乘积
        int[] maxDp = new int[len];
        // minDp[i]：以 num[i] 为结尾的连续子数组最小乘积，为了记录两个负数时的情况
        int[] minDp = new int[len];
        maxDp[0] = nums[0];
        minDp[0] = nums[0];
        for (int i = 1; i < len; i++) {
            maxDp[i] = Math.max(nums[i] * maxDp[i - 1], Math.max(nums[i], nums[i] * minDp[i - 1]));
            minDp[i] = Math.min(nums[i] * minDp[i - 1], Math.min(nums[i], nums[i] * maxDp[i - 1]));
        }

        int res = Integer.MIN_VALUE;
        for (int num : maxDp) {
            res = Math.max(res, num);
        }
        return res;
    }

    // 空间压缩
    public static int maxProduct2(int[] nums) {
        int len = nums.length;
        if (len == 0) return 0;

        int preMax = nums[0];
        int preMin = nums[0];
        int res = Integer.MIN_VALUE;
        for (int i = 1; i < len; i++) {
            int max = Math.max(nums[i] * preMax, Math.max(nums[i], nums[i] * preMin));
            int min = Math.min(nums[i] * preMin, Math.min(nums[i], nums[i] * preMax));
            preMax = max;
            preMin = min;
            res = Math.max(res, max);
        }
        return res;
    }

    //==================================================================================================================

    /**
     * 169. 多数元素
     * 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
     * 输入：nums = [3,2,3]
     * 输出：3
     * 尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。
     */
    public static int majorityElement(int[] nums) {
        // 众数是要 > n/2，所以可以用 count 记录碰到众数的加一，不是就减一，最终count会大等于1
        int count = 0;
        Integer candidate = null;

        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }

            count += (candidate == num) ? 1 : -1;
        }

        return candidate;
    }


    //==================================================================================================================

    /**
     * 198. 打家劫舍
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     * <p>
     * 输入：[1,2,3,1]
     * 输出：4
     * 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     * 偷窃到的最高金额 = 1 + 3 = 4 。
     */
    public static int rob(int[] nums) {
        robMemo = new int[nums.length];
        Arrays.fill(robMemo, -1);
        return dp(nums, 0);
    }

    private static int[] robMemo;

    // 返回 dp[i..] 能抢到的最大值
    private static int dp(int[] nums, int i) {
        if (i >= nums.length) return 0;
        if (robMemo[i] != -1) return robMemo[i];
        // 打劫 or 不打劫
        robMemo[i] = Math.max(nums[i] + dp(nums, i + 2), dp(nums, i + 1));
        return robMemo[i];

    }

    /**
     * 213. 打家劫舍 II
     * 这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的
     * <p>
     * 输入：nums = [2,3,2]
     * 输出：3
     * 解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
     */
    public static int rob2(int[] nums) {

        // 三种不同情况：要么都不被抢；要么第一间房子被抢最后一间不抢；要么最后一间房子被抢第一间不抢。
        // 情况一的结果肯定最小，只要比较情况二和情况三就行了
        int len = nums.length;
        if (len == 1) return nums[0];
        int[] memo1 = new int[len];
        int[] memo2 = new int[len];

        Arrays.fill(memo1, -1);
        Arrays.fill(memo2, -1);

        return Math.max(dp(nums, 0, len - 2, memo1), dp(nums, 1, len - 1, memo2));

    }

    // 返回 dp[i..j] 能抢到的最大值
    private static int dp(int[] nums, int i, int j, int[] memo) {
        if (i > j) return 0;

        if (memo[i] != -1) return memo[i];

        memo[i] = Math.max(nums[i] + dp(nums, i + 2, j, memo), dp(nums, i + 1, j, memo));

        return memo[i];
    }


    /**
     * 337. 打家劫舍 III
     * 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，称之为root。
     * 除了root之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
     * 给定二叉树的root。返回在不触动警报的情况下，小偷能够盗取的最高金额。
     * <p>
     * 输入: root = [3,2,3,null,3,null,1]
     * 输出: 7
     * 解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7
     */
    public static int rob3(TreeNode root) {
        if (root == null) return 0;
        return dp(root, true);
    }

    private static HashMap<TreeNode, Integer> canRobMemo = new HashMap<>();
    private static HashMap<TreeNode, Integer> notRobMemo = new HashMap<>();

    private static int dp(TreeNode root, boolean isCanRob) {
        if (root == null) return 0;
        if (isCanRob && canRobMemo.containsKey(root)) return canRobMemo.get(root);
        if (!isCanRob && notRobMemo.containsKey(root)) return notRobMemo.get(root);
        int res;
        if (isCanRob) {
            // 抢 or 不抢
            res = Math.max(root.val + dp(root.left, false) + dp(root.right, false), dp(root.left, true) + dp(root.right, true));
            canRobMemo.put(root, res);
        } else {
            res = dp(root.left, true) + dp(root.right, true);
            notRobMemo.put(root, res);
        }

        return res;
    }

    // 不用双备忘录
    public static int rob4(TreeNode root) {
        if (root == null) return 0;
        if (robMemo3.containsKey(root)) return robMemo3.get(root);

        // 抢 or 不抢
        int do_it = root.val + (root.left == null ? 0 : rob4(root.left.left) + rob4(root.left.right)) + (root.right == null ? 0 : rob4(root.right.left) + rob4(root.right.right));
        int do_no = rob4(root.left) + rob4(root.right);

        int res = Math.max(do_it, do_no);
        robMemo3.put(root, res);

        return res;
    }

    private static HashMap<TreeNode, Integer> robMemo3 = new HashMap<>();


    //==================================================================================================================


    /**
     * 200. 岛屿数量
     * 给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 此外，你可以假设该网格的四条边均被水包围。
     */
    public static int numIslands(char[][] grid) {
        int res = 0;
        int m = grid.length, n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    // 每发现一个岛屿数量加一
                    res++;
                    // 并把岛屿淹没
                    dfs(grid, i, j);
                }
            }
        }
        return res;
    }

    // 用 DFS 算法解决岛屿题目是最常见的，每次遇到一个岛屿中的陆地，就用 DFS 算法吧这个岛屿「淹掉」。避免维护 visited 数组。
    // 从（i, j）开始，将与之相邻的陆地都变成水
    private static void dfs(char[][] grid, int i, int j) {
        int m = grid.length, n = grid[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n) return;
        if (grid[i][j] == '0') return; // 已经是水了

        // 将(i,j)变成海水
        grid[i][j] = '0';
        // 淹没上下左右的陆地
        dfs(grid, i + 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i - 1, j);
        dfs(grid, i, j - 1);
    }

    /**
     * 1254. 统计封闭岛屿的数目
     * 二维矩阵 grid由 0（土地）和 1（水）组成。岛是由最大的4个方向连通的 0组成的群，封闭岛是一个完全 由1包围（左、上、右、下）的岛。
     * 请返回 封闭岛屿 的数目。
     */
    public static int closedIsland(int[][] grid) {
        // 思路：和上面 200 题差不多，把靠边的岛屿排除掉，剩下的就是「封闭岛屿」
        int m = grid.length, n = grid[0].length;
        for (int i = 0; i < m; i++) {
            // 排除左右靠边
            dfs(grid, i, 0);
            dfs(grid, i, n - 1);
        }
        for (int j = 0; j < n; j++) {
            // 排除上下靠边
            dfs(grid, 0, j);
            dfs(grid, m - 1, j);
        }

        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    res++;
                    dfs(grid, i, j);
                }
            }
        }
        return res;

    }

    private static void dfs(int[][] grid, int i, int j) {
        int m = grid.length, n = grid[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n) return;
        if (grid[i][j] == 1) return;
        grid[i][j] = 1;
        dfs(grid, i - 1, j);
        dfs(grid, i + 1, j);
        dfs(grid, i, j - 1);
        dfs(grid, i, j + 1);
    }

    /**
     * 1020. 飞地的数量
     * 给你一个大小为 m x n 的二进制矩阵 grid ，其中 0 表示一个海洋单元格、1 表示一个陆地单元格。
     * 一次 移动 是指从一个陆地单元格走到另一个相邻（上、下、左、右）的陆地单元格或跨过 grid 的边界。
     * 返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。
     */
    public static int numEnclaves(int[][] grid) {

        // 和上面 1254 题目类似，多了个计算数量而已
        return 0;
    }

    /**
     * 695. 岛屿的最大面积
     * 给你一个大小为 m x n 的二进制矩阵 grid 。
     * 岛屿是由一些相邻的1(代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设grid 的四个边缘都被 0（代表水）包围着
     * 岛屿的面积是岛上值为 1 的单元格的数目。
     * 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
     */
    public int maxAreaOfIsland(int[][] grid) {
        // 和上面 200 题目类似，多了个计算最大数量而已
        return 0;
    }

    /**
     * 1905. 统计子岛屿
     * 给你两个m x n的二进制矩阵grid1 和grid2，它们只包含0（表示水域）和 1（表示陆地）。一个 岛屿是由 四个方向（水平或者竖直）上相邻的1组成的区域。任何矩阵以外的区域都视为水域。
     * 如果 grid2的一个岛屿，被 grid1的一个岛屿完全 包含，也就是说 grid2中该岛屿的每一个格子都被 grid1中同一个岛屿完全包含，那么我们称 grid2中的这个岛屿为 子岛屿。
     * 请你返回 grid2中 子岛屿的 数目。
     */
    public static int countSubIslands(int[][] grid1, int[][] grid2) {
        // 和上面 200 题目类似，以 grid2 来淹没岛屿，计算子岛屿数量
        // 思路：如果岛屿 B 中存在一片陆地，在岛屿 A 的对应位置是海水，那么岛屿 B 就不是岛屿 A 的子岛。
        return 0;
    }

    /**
     * 694. 不同的岛屿数量
     * 输入一个二维矩阵，0 表示海水，1 表示陆地，这次让你计算 不同的 (distinct) 岛屿数量
     */
    public static int numDistinctIslands(int[][] grid) {
        // 思路：想办法把二维矩阵中的「岛屿」进行转化，变成比如字符串这样的类型，然后利用 HashSet 这样的数据结构去重，最终得到不同的岛屿的个数。
        int m = grid.length, n = grid[0].length;
        // 记录所有岛屿的序列化结果
        HashSet<String> islands = new HashSet<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    StringBuilder sb = new StringBuilder();
                    dfs(grid, i, j, sb, 666);// 初始的方向可以随便写，不影响正确性
                    islands.add(sb.toString());
                }
            }
        }
        return islands.size();
    }

    // 淹没岛屿，并且序列化遍历顺序（相同的岛屿遍历顺序一样）
    private static void dfs(int[][] grid, int i, int j, StringBuilder sb, int dir) {
        int m = grid.length, n = grid[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0) return;

        grid[i][j] = 0;
        sb.append(dir).append(','); // 前序遍历位置：进入 (i, j)
        // 上下左右
        dfs(grid, i - 1, j, sb, dir);
        dfs(grid, i + 1, j, sb, dir);
        dfs(grid, i, j - 1, sb, dir);
        dfs(grid, i, j + 1, sb, dir);
        sb.append(-dir).append(',');// 后序遍历位置：离开 (i, j)
    }


    //==================================================================================================================

    public static void main(String[] args) {
        // [4,2,1,3]
        ListNode head = new ListNode(4);
        ListNode p = head;
        p.next = new ListNode(2);
        p = p.next;
        p.next = new ListNode(1);
        p = p.next;
        p.next = new ListNode(3);

        print(head);

        ListNode sorted = sortList(head);
        print(sorted);

    }

    public static void print(ListNode node) {
        System.out.print("[ ");
        ListNode p = node;
        while (p != null) {
            System.out.print(p.val);
            System.out.print(" ");
            p = p.next;
        }
        System.out.print("]");
    }

    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
}
