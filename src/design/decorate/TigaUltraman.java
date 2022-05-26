package design.decorate;

/**
 * 4. 具体装饰者
 */
public class TigaUltraman extends Master {

    public TigaUltraman(Superman man) {
        super(man);
    }

    public void teachAttack() {
        System.out.println("TigaUltraman teachAttack");
    }

    @Override
    public void attack() {
        super.attack();
        teachAttack();
    }
}
