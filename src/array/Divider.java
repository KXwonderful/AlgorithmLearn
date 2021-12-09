package array;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

/**
 * 分治算法
 */
public class Divider {

    /**
     * 典型的分治算法：归并排序
     */
    void sort(int[] nums, int low, int high) {
        int mid = (low + high) / 2;

        // ------ 分 -----
        sort(nums, low, mid);
        sort(nums, mid, high);
        // ------ 治 ------

        // 合并
        //merge(nums, low, mid, high);
    }

    /**
     * 力扣 241 题，添加括号的所有方式
     * <p>
     * 输入一个算式，你可以给它随意加括号，请你穷举出所有可能的加括号方式，并计算出对应的结果
     * <p>
     * 分治思想，以每个运算符作为分割点，把复杂问题分解成小的子问题，递归求解子问题，然后再通过子问题的结果计算出原问题的结果
     */
    List<Integer> diffWaysToCompute(String input) {
        // 避免重复计算
        if (memo.containsKey(input)) return memo.get(input);

        List<Integer> res = new LinkedList<>();
        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            // 扫描算式 input 中的运算符
            if (c == '-' || c == '*' || c == '+') {
                /* ***** 分 ******/
                // 以运算符为中心，分割成两个字符串，分别递归计算
                List<Integer> left = diffWaysToCompute(input.substring(0, i));
                List<Integer> right = diffWaysToCompute(input.substring(i + 1));
                /* ***** 治 ******/

                // 通过子问题的结果，合成原问题的结果
                for (int a : left) {
                    for (int b : right) {
                        if (c == '+') res.add(a + b);
                        else if (c == '-') res.add(a - b);
                        else if (c == '*') res.add(a * b);
                    }
                }
            }
        }

        // base case : 如果 res 为空，说明算式是一个数字，没有运算符
        if (res.isEmpty()) res.add(Integer.parseInt(input));

        // 添加备忘录
        memo.put(input, res);

        return res;
    }

    // 备忘录
    HashMap<String, List<Integer>> memo = new HashMap<>();
}
