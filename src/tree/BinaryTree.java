package tree;


/**
 * 二叉树
 */
public class BinaryTree {

    /**
     * 输入一棵普通二叉树，返回节点总数
     * <p>
     * 时间复杂度 O(N)
     */
    int countNodes1(TreeNode root) {
        if (root == null) return 0;
        return 1 + countNodes1(root.left) + countNodes1(root.right);
    }

    /**
     * 输入一棵满二叉树，返回节点总数
     * <p>
     * 时间复杂度 O(logN)
     */
    int countNodes2(TreeNode root) {
        int high = 0; // 二叉树高度
        while (root.left != null) {
            high++;
            root = root.left;
        }
        // 节点总数就是 2^h - 1
        return (int) Math.pow(2, high) - 1;
    }

    /**
     * 输入一棵完全二叉树，返回节点总数
     * <p>
     * 时间复杂度 O(logN*logN)
     */
    int countNodes3(TreeNode root) {
        if (root == null) return 0;
        TreeNode l = root, r = root;
        // 记录左右子树高度
        int hl = 0, hr = 0;
        while (l != null) {
            l = l.left;
            hl++;
        }
        while (r != null) {
            r = r.right;
            hr++;
        }
        // 如果左右子树的高度相同，则是一棵满二叉树
        if (hl == hr) {
            return (int) Math.pow(2, hl) - 1;
        }
        // 如果左右高度不同，则按照普通二叉树的逻辑计算
        return 1 + countNodes3(root.left) + countNodes3(root.right);
    }


    /**
     * 基本二叉树节点
     */
    static class TreeNode {
        int val;
        TreeNode left, right;

        public TreeNode(int val) {
            this.val = val;
        }
    }
}
