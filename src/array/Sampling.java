package array;

import java.util.LinkedList;
import java.util.Random;

/**
 * 水塘抽样算法（Reservoir Sampling）
 */
public class Sampling {

    public static void main(String[] args) {
        // test
        int n = 10;
        // 答案：4
        //System.out.println("res=" + countPrimes(n));
    }


    /**
     * 382. 链表随机节点
     * <p>
     * 给你一个未知长度的链表，请你设计一个算法，只能遍历一次，随机地返回链表中的一个节点。
     * <p>
     * 这里说随机是均匀随机（uniform random），若有n个元素，每个元素被选中的概率都是1/n。
     */
    public static void solution(ListNode head) {

    }

    /**
     * 返回链表中一个随机节点的值
     * <p>
     * 当遇到第i个元素时，应该有1/i的概率选择该元素，1 - 1/i的概率保持原有的选择。
     */
    public static int getRandom(ListNode head) {
        Random r = new Random();
        int i = 0, res = 0;
        ListNode p = head;
        while (p != null) {
            // 生成一个 [0, i) 之间的整数，这个整数等于 0 的概率就是 1/i
            if (r.nextInt(++i) == 0) {
                res = p.val;
            }
            p = p.next;
        }
        return res;
    }

    /**
     * 返回链表中 k 个随机节点的值
     * <p>
     * 若要随机选择k个数，只要在第i个元素处以k/i的概率选择该元素，以1 - k/i的概率保持原有选择即可
     */
    public static int[] getRandom(ListNode head, int k) {
        Random r = new Random();
        int[] res = new int[k];
        ListNode p = head;

        // 前 k 个元素先默认选上
        for (int j = 0; j < k && p != null; j++) {
            res[j] = p.val;
            p = p.next;
        }

        int i = k;
        // while 循环遍历链表
        while (p != null) {
            // 生成一个 [0, i) 之间的整数
            int j = r.nextInt(++i);
            // 这个整数小于 k 的概率就是 k/i
            if (j < k) {
                res[j] = p.val;
            }
            p = p.next;
        }
        return res;
    }

    /**
     * 398. 随机数索引
     *
     * 给定一个可能含有重复元素的整数数组，要求随机输出给定的数字的索引。 您可以假设给定的数字一定存在于数组中。
     */
    public static int pick(int[] nums, int target) {
        LinkedList<Integer> indexes = new LinkedList<>();
        for (int i = 0; i < nums.length ;i++){
            if (nums[i] == target){
                indexes.add(i);
            }
        }

        Random r = new Random();
        int i = 0, res = indexes.get(0);
        while (i < indexes.size()){
            // 生成一个 [0, i) 之间的整数，这个整数等于 0 的概率就是 1/i
            if (r.nextInt(++i) == 0){
                res = indexes.get(i-1);
            }
        }
        return res;
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
}
