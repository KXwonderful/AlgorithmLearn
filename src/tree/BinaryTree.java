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

    // *****************************  二叉树的最近公共祖先  ******************************

    /**
     * 二叉树的最近公共祖先
     * <p>
     * root节点确定了一棵二叉树，p 和 q 是这这棵二叉树上的两个节点，
     * 返回 p 节点和 q 节点的最近公共祖先节点
     */
    TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

        //情况 1，如果p和q都在以root为根的树中，函数返回的即使p和q的最近公共祖先节点。
        //情况 2，那如果p和q都不在以root为根的树中怎么办呢？函数理所当然地返回null呗。
        //情况 3，那如果p和q只有一个存在于root为根的树中呢？函数就会返回那个节点。

        // base case
        if (root == null) return null;
        if (root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        // 情况1
        if (left != null && right != null) return root;

        // 情况2
        if (left == null && right == null) return null;
        // 情况3
        return left == null ? right : left;
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
