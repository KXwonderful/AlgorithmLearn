package other;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class BacktrackTest {

    public static void main(String[] args) {

    }

    /**
     * 矩阵中的路径
     */
    boolean exist(char[][] board, String word) {
        char[] words = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (backtrack(board, words, 0, 0, 0)) return true;
            }
        }
        return false;
    }

    /**
     * 当前元素在矩阵 board 中的行列索引 i 和 j ，当前目标字符在 word 中的索引 k
     */
    boolean backtrack(char[][] board, char[] words, int i, int j, int k) {
        if (i < 0 || i > board.length - 1 || j < 0 || j > board[0].length - 1 || board[i][j] != words[k]) return false;
        if (k == words.length - 1) return true;
        board[i][j] = '0';
        boolean res = backtrack(board, words, i + 1, j, k + 1) || backtrack(board, words, i - 1, j, k + 1) || backtrack(board, words, i, j + 1, k + 1) || backtrack(board, words, i, j - 1, k + 1);
        board[i][j] = words[k];
        return res;
    }

    /**
     * 全排列：输入一组不重复的数字，返回它们的全排列
     * <p>
     * 时间复杂度：O(N!)
     */
    public List<List<Integer>> permute(int[] nums) {
        LinkedList<Integer> track = new LinkedList<>();
        backtrack(nums, track);
        return res;
    }

    private final List<List<Integer>> res = new LinkedList<>();

    void backtrack(int[] nums, LinkedList<Integer> track) {
        if (track.size() == nums.length) {
            res.add(new LinkedList<>(track));
        }
        for (int num : nums) {
            // 排除不合法的选择
            if (track.contains(num)) continue;
            // 做选择
            track.add(num);
            backtrack(nums, track);
            // 撤销选择
            track.removeLast();
        }
    }

    /**
     * N 皇后问题：输入棋盘边长 n，返回所有合法的放置
     */
    public List<List<String>> solveNQueens(int n) {
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        backtrack(board, 0);
        return res2;
    }

    List<List<String>> res2 = new ArrayList<>();

    void backtrack(char[][] board, int row) {
        if (row == board.length) {
            res2.add(convertToList(board));
            return;
        }

        int n = board[0].length;
        for (int col = 0; col < n; col++) {
            if (!isValid(board, row, col)) continue;

            board[row][col] = 'Q';
            backtrack(board, row + 1);
            board[row][col] = '.';
        }
    }

    boolean isValid(char[][] board, int row, int col) {
        int n = board[0].length;
        // 列
        for (int i = 0; i < n; i++) {
            if (board[i][col] == 'Q') return false;
        }

        // 右上方
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') return false;
        }

        // 左上方
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') return false;
        }

        return true;

    }

    /**
     * char[][] 转成 List<String>
     */
    List<String> convertToList(char[][] board) {
        List<String> list = new ArrayList<>();
        for (char[] chars : board) {
            list.add(new String(chars));
        }
        return list;
    }

    /**
     * 括号生成
     */
    public List<String> generateParenthesis(int n) {
        List<Character> track = new ArrayList<>();
        backtrack(n, n, track);
        return res3;
    }

    List<String> res3 = new ArrayList<>();

    // 可用的左括号数量为 left 个，可用的右括号数量为 right 个
    void backtrack(int left, int right, List<Character> track) {
        if (right < left) return;
        if (left < 0 || right < 0) return;
        if (left == 0 && right == 0) {
            StringBuilder sb = new StringBuilder();
            for (Character c : track) {
                sb.append(c);
            }
            res3.add(sb.toString());
            return;
        }

        track.add('(');
        backtrack(left - 1, right, track);
        track.remove(track.size() - 1);

        track.add(')');
        backtrack(left, right - 1, track);
        track.remove(track.size() - 1);
    }

    /**
     * 子集
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<Integer> track = new ArrayList<>();
        backtrack(nums, 0, track);
        return res4;
    }

    List<List<Integer>> res4 = new ArrayList<>();

    void backtrack(int[] nums, int start, List<Integer> track) {
        res4.add(new ArrayList<>(track));
        for (int i = start; i < nums.length; i++) {
            track.add(nums[i]);
            backtrack(nums, i + 1, track);
            track.remove(track.size() - 1);
        }
    }

    /**
     * 组合
     */
    public List<List<Integer>> combine(int n, int k) {
        if (k <= 0 || n <= 0) return res5;
        List<Integer> track = new ArrayList<>();
        backtrack(n, k, 1, track);
        return res5;
    }

    List<List<Integer>> res5 = new ArrayList<>();

    void backtrack(int n, int k, int start, List<Integer> track) {
        if (k == track.size()) {
            res5.add(new ArrayList<>(track));
            return;
        }

        for (int i = start; i <= n; i++) {
            track.add(i);
            backtrack(n, k, i + 1, track);
            track.remove(track.size() - 1);
        }
    }


    /**
     * 机器人的运动范围
     */
    public int movingCount(int m, int n, int k) {
        visited = new boolean[m][n];
        this.m = m;
        this.n = n;
        return backtrack(0, 0, k);
    }

    boolean[][] visited;
    int m, n;

    int backtrack(int i, int j, int k) {
        // i >= m || j >= n是边界条件的判断，visited[i][j]判断这个格子是否被访问过
        // k < sum(i, j)判断当前格子坐标是否满足条件
        if (i >= m || j >= n || visited[i][j] || sums(i, j) > k) return 0;
        // 标注这个格子被访问过
        visited[i][j] = true;
        // 沿着当前格子的右边和下边继续访问
        return 1 + backtrack(i + 1, j, k) + backtrack(i, j + 1, k);
    }

    // 计算两个坐标数字的和
    int sums(int i, int j) {
        int sum = 0;
        while (i != 0) {
            sum += i % 10;
            i = i / 10;
        }
        while (j != 0) {
            sum += j % 10;
            j = j / 10;
        }
        return sum;
    }

    /**
     * 和为s的连续正数序列
     */
    public int[][] findContinuousSequence(int target) {
        if (target <= 1) return new int[0][0];
        backtrackSequence(1, target);
        int[][] res = resSequence.stream().map(list -> list.stream().mapToInt(i -> i).toArray()).toArray(int[][]::new);
        return res;
    }

    List<List<Integer>> resSequence = new ArrayList<>();

    void backtrackSequence(int start, int target) {
        if (start >= target) return;

        int sum = start;
        int index = start;
        while (sum < target) {
            index++;
            sum += index;
        }

        if (sum == target) {
            List<Integer> list = new ArrayList<>();
            for (int i = start; i < index; i++) {
                list.add(i);
            }
            resSequence.add(list);
        }

        backtrackSequence(++start, target);
    }

}
