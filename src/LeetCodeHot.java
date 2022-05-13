import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

public class LeetCodeHot {

    public static void main(String[] args) {

    }

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

    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
