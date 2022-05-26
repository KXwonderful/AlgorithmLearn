package design.decorate;

/**
 * 3. 抽象装饰者
 */
public abstract class Master extends Superman {
    private Superman superman;

    public Master(Superman man) {
        superman = man;
    }

    @Override
    public void attack() {
        superman.attack();
    }
}
