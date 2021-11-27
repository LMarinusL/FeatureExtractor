using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Globalization;

public class PCGenerator : MonoBehaviour
{

    public TextAsset PCFile;
    public GameObject dot;
    public List<Vector3> vertices = null;



    void Start()
    {
        ReadFile();
        AddPoints();
    }
    void ReadFile()
    {
        string PointsString = PCFile.ToString();
        string[] arrayOfLines = PointsString.Split('\n');
        int index = 0;
        string[] values;
        Vector3 VectorNew;
        while (index < arrayOfLines.Length - 1)
        {
            values = arrayOfLines[index].Split(' ');
            VectorNew = new Vector3(((float.Parse(values[1], CultureInfo.InvariantCulture) - 1013618) / 10),
                          ((float.Parse(values[2], CultureInfo.InvariantCulture) - 265) / 10),
                          ((float.Parse(values[0], CultureInfo.InvariantCulture) - 649582) / 10));
            vertices.Add(VectorNew);
            index++;
        }




    }
            
    void AddPoints()
    {
    for (int vertId = 0; vertId < vertices.ToArray().Length; vertId++){
            Instantiate(dot, vertices[vertId], transform.rotation);
        }
    }
}
