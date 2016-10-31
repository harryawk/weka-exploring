/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil.dua.ai;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Toshiba
 */
public class TucilDuaAi {
    
    static Instances datas;
    
    public static void LoadData() throws Exception {
        datas = DataSource.read
        ("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
        datas.setClassIndex(datas.numAttributes()-1); //Set label atribute
    }
    
    public static Instances Discretize() throws Exception {
        Discretize discretize = new Discretize();
        String[] options = new String[2];
        options[0] = "-R";                                    
        options[1] = "1";                                     
        discretize.setOptions(options);
        discretize.setInputFormat(datas);
        return Filter.useFilter(datas,discretize);
    }
    
    public static void main(String[] args) {
        // TODO code application logic here
        TucilDuaAi test= new TucilDuaAi();
        try {
            LoadData();
            Instances temp = Discretize();
            System.out.println(temp);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
}
