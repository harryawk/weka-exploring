/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil.dua.ai;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

/**
 *
 * @author Toshiba
 */
public class TucilDuaAi {
    
    static Instances datas;
    
    public static void LoadData() throws Exception {
        datas = DataSource.read("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
//        System.out.println(datas);
    }
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        try {
            LoadData();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(datas);
    }
    
}
