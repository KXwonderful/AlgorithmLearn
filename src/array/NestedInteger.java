package array;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * 数据结构NestedInteger，这个结构中存的数据可能是一个Integer整数，也可能是一个NestedInteger列表
 */
public class NestedInteger {

    private Integer val;
    private List<NestedInteger> list;

    public NestedInteger(int val) {
        this.val = val;
        this.list = null;
    }

    public NestedInteger(List<NestedInteger> list) {
        this.val = null;
        this.list = list;
    }

    // 如果其中存的是一个整数，则返回 true，否则返回 false
    public boolean isInteger() {
        return val != null;
    }

    // 如果其中存的是一个整数，则返回这个整数，否则返回 null
    public Integer getInteger() {
        return this.val;
    }

    // 如果其中存的是一个列表，则返回这个列表，否则返回 null
    public List<NestedInteger> getList() {
        return this.list;
    }

    /**
     * 遍历 N 叉树 的所有叶子节点
     */
    static class NestedIterator implements Iterator<Integer> {

        private Iterator<Integer> it;

        public NestedIterator(List<NestedInteger> nestedList) {
            // 存放将 nestedList 打平的结果
            List<Integer> result = new LinkedList<>();
            for (NestedInteger node : nestedList) {
                // 以每个节点为根遍历
                traverse(node, result);
            }
            // 得到 result 列表的迭代器
            this.it = result.iterator();
        }

        @Override
        public Integer next() {
            return it.next();
        }

        @Override
        public boolean hasNext() {
            return it.hasNext();
        }

        // 遍历以 root 为根的多叉树，将叶子节点的值加入 result 列表
        private void traverse(NestedInteger root, List<Integer> result) {
            if (root.isInteger()) {
                // 到达叶子节点
                result.add(root.getInteger());
                return;
            }
            // 遍历框架
            for (NestedInteger child : root.getList()) {
                traverse(child, result);
            }
        }
    }

    /**
     * 优化：
     * 调用hasNext时，如果nestedList的第一个元素是列表类型，则不断展开这个元素，直到第一个元素是整数类型
     */
    static class NestedIterator2 implements Iterator<Integer> {

        private LinkedList<NestedInteger> list;

        public NestedIterator2(List<NestedInteger> nestedList) {
            // 不直接用 nestedList 的引用，是因为不能确定它的底层实现
            // 必须保证是 LinkedList，否则下面的 addFirst 会很低效
            list = new LinkedList<>(nestedList);
        }

        @Override
        public Integer next() {
            // hasNext 方法保证了第一个元素一定是整数类型
            return list.remove(0).getInteger();
        }

        @Override
        public boolean hasNext() {
            // 循环拆分列表元素，直到列表第一个元素是整数类型
            while (!list.isEmpty() && !list.get(0).isInteger()) {
                // 当列表开头第一个元素是列表类型时，进入循环
                List<NestedInteger> first = list.remove(0).getList();
                // 将第一个列表打平并按顺序添加到开头
                for (int i = first.size() - 1; i >= 0; i--) {
                    list.addFirst(first.get(i));
                }
            }
            return !list.isEmpty();
        }

    }

}
