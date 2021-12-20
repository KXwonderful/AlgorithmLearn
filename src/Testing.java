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
        for(int i = 1; i < array.length; i++){
            int temp = array[i];// 记录要插入的数据
            // 从已经排序的序列最右边的开始比较，找到比其小的数
            int j = i;
            while (j > 0 && array[j-1] > temp){
                array[j] = array[j-1];
                j--;
            }
            if (j != i) array[j] = temp;
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
