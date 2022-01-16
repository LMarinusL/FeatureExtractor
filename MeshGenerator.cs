using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Globalization;
using TMPro;
using UnityEngine.UI;

//[ExecuteInEditMode] // this loads terrain in edit mode
/* to do: 
 - Set automatic mesh creation to file reader
 - Add file writer
 - create component class for output grid
 - Create grid to assign properties
 - Create component class for medial balls to save properties
 - Filter the 2D set
 - Create component for medial branches
 - Compute medial branches 
*/


public class MeshGenerator : MonoBehaviour
{
    public TextAsset vertexFile;
    public TextAsset triangleFile;
    [SerializeField] public TextMeshProUGUI loadingText;


    public Mesh mesh;
    public List<Vector3> vertices = null;
    public List<int> triangles;
    public List<string> idList;
    public float heightScale = 5.0f;

    public Gradient gradient;
    public float maxTerrainHeight;
    public float minTerrainHeight;
    

    Color[] colors;
    char[] charsToTrim = { '*', ' ', '\n', '\r' };


    void Start()
    {
        mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;
        ReadFile();
        UpdateMesh();
    }
    void ReadFile()
    {
        string PointsString = vertexFile.ToString();
        string[] arrayOfLines = PointsString.Split('\n');
        int index = 0;
        string[] values;
        Vector3 VectorNew;
        while (index < arrayOfLines.Length -1)
        {
            values = arrayOfLines[index].Split(' ');
            VectorNew = new Vector3(((float.Parse(values[1], CultureInfo.InvariantCulture)- 1013618) / 10),
                               ((float.Parse(values[2], CultureInfo.InvariantCulture)) / 10),
                               ((float.Parse(values[0], CultureInfo.InvariantCulture)- 649582) / 10));
            vertices.Add(VectorNew);
            idList.Add(values[3].Trim(charsToTrim)); // list in order with vertex id's
            if (VectorNew.y > maxTerrainHeight)
            {
                maxTerrainHeight = VectorNew.y;
            }
            if (VectorNew.y < minTerrainHeight)
            {
                minTerrainHeight = VectorNew.y;
            }
            index++;
        }

        string TrianglesString = triangleFile.ToString();
        string[] arrayOfLines2 = TrianglesString.Split('\n');
        int indexT = 0;
        string[] valuesT;
        while (indexT < arrayOfLines2.Length-1 )
        {
            valuesT = arrayOfLines2[indexT].Split(' ');
            triangles.Add(idList.IndexOf(valuesT[0]));
            triangles.Add(idList.IndexOf(valuesT[1]));
            triangles.Add(idList.IndexOf(valuesT[2].Trim(charsToTrim)));
            indexT++;
        }

        colors = new Color[vertices.ToArray().Length];
        for ( int i = 0; i < vertices.ToArray().Length; i++)
        {
            float height = Mathf.InverseLerp(minTerrainHeight, maxTerrainHeight,  vertices[i].y);
            colors[i] = gradient.Evaluate(height);
        }
    }

   public  void AdjustScale(float newScale) {    
        transform.localScale += new Vector3(0, newScale, 0);
    }

    void UpdateMesh()
    {
 
        mesh.Clear();
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.colors = colors;
        mesh.RecalculateNormals();
    }
}
