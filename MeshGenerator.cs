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
 - Remake old params
 - Make relative height, slope and aspect params
 - Filter the 2D set
 - Create component for medial branches
 - Compute medial branches 
*/


public class MeshGenerator : MonoBehaviour
{
    public TextAsset vertexFile;
    [SerializeField] public TextMeshProUGUI loadingText;

    public int zSizer; //original full 369 small 188
    public int xSizer; // original full 752 small 93
    int zSize;
    int xSize;
    public Mesh mesh;
    public List<Vector3> vertices = null;
    public int[] triangles;
    public List<string> idList;
    public float heightScale = 5.0f;

    public float maxTerrainHeight;
    public float minTerrainHeight;
    public float xCorrection = 649582f;
    public float zCorrection = 1013618f;



    Color[] colors;
    public Color color1 = new Color(0f, 0.65f, 0.95f, 95f);
    public Color color2 = new Color(0f, 0.6f, 0.0f, 0.6f);
    public Color color3 = new Color(0.8f, 0.8f, 0.8f, 1f);
    public Material material;

    char[] charsToTrim = { '*', ' ', '\n', '\r' };


    void Start()
    {
        mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;
        MeshRenderer meshr = this.GetComponent<MeshRenderer>();
        meshr.material = material;
        ReadFile();
        UpdateMesh();
    }

    void Update()
    {
        /*
        if (Input.GetKey(KeyCode.Alpha1))
        {
            setMeshColors();
        }*/
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
            VectorNew = new Vector3(((float.Parse(values[1], CultureInfo.InvariantCulture)- zCorrection) / 10),
                               ((float.Parse(values[2], CultureInfo.InvariantCulture)) / 3),
                               ((float.Parse(values[0], CultureInfo.InvariantCulture)- xCorrection) / 10));
            vertices.Add(VectorNew);
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

        zSize = zSizer - 1;
        xSize = xSizer - 1;
        triangles = new int[xSize * zSize * 6];
        int vert = 0;
        int tris = 0;
        for (int z = 0; z < zSize; z++)
        {
            for (int x = 0; x < xSize; x++)
            {
                triangles[tris + 0] = vert + 0;
                triangles[tris + 1] = vert + 1;
                triangles[tris + 2] = vert + xSize + 1;
                triangles[tris + 3] = vert + 1;
                triangles[tris + 4] = vert + xSize + 2;
                triangles[tris + 5] = vert + xSize + 1;

                vert++;
                tris += 6;

            };
            vert++;
        }
        colors = new Color[vertices.ToArray().Length];
        for (int i = 0; i < vertices.ToArray().Length; i++)
        {
            if (vertices[i].y < 210)
            {
                colors[i] = color2;
            }
            else { colors[i] = color3; }
        }

    }
    /*
    void setMeshColors()
    {
        Vector3[] normals = mesh.normals;
        colors = new Color[vertices.ToArray().Length];
        for (int i = 0; i < vertices.ToArray().Length; i++)
        {
           colors[i] = new Color(1f* normals[i].y, 0f, 1f*(1- normals[i].y), 1f);
        }
        mesh.colors = colors;
    }*/

    public  void AdjustScale(float newScale) {    
        transform.localScale += new Vector3(0, newScale, 0);
    }

    void UpdateMesh()
    {
        mesh.Clear();
        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles;
        mesh.colors = colors;
        mesh.RecalculateNormals();
    }
}
