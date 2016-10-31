/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil.dua.ai;

import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.Instance;

/**
 *
 * @author Toshiba
 */
public class TucilDuaAi {
    
    static Instances datas;
    /**
     * 
     */
    public static void Classifier() {
       J48 tree = new J48();
       String[] options = new String[2];
       
       
    }
    /**
     * 
     * @throws Exception 
     */
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
//        System.out.println(datas.get(1).dataset());
        double[] dabel = datas.instance(0).toDoubleArray();
//        System.out.println(datas.instance(0).toDoubleArray());
        Instance data;
        for (int i = 0; i < dabel.length; i++) {
            System.out.println(dabel[i]);
        }
        
//        System.out.println(datas.instance(0));
    }
    
}
