import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
        for (int i = 2; i <= n; i++){
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
