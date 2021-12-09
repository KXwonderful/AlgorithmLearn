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
