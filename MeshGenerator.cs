using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Globalization;
using TMPro;
using UnityEngine.UI;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;


public class MeshGenerator : MonoBehaviour
{
    public TextAsset vertexFile;
    [SerializeField] public TextMeshProUGUI loadingText;

    public int zSizer; //original full 752 small 188
    public int xSizer; // original full 369 small 93
    int zSize;
    int xSize;
    public Mesh mesh;
    public float3[] vertices;
    public float3[] normals;
    public int[] triangles;
    public List<string> idList;
    public float heightScale = 5.0f;

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
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        GetComponent<MeshFilter>().mesh = mesh;
        MeshRenderer meshr = this.GetComponent<MeshRenderer>();
        meshr.material = material;
        ReadFile();
        UpdateMesh();
    }

    void Update()
    {
        
        if (Input.GetKey(KeyCode.L))
        {
            CreateShape();
            UpdateMesh();
        }

    }

    void CreateShape()
    {
        float randomNum1 = UnityEngine.Random.Range(0.3f, 2.4f);
        float randomNum2 = UnityEngine.Random.Range(0.3f, 2.4f);
        vertices = new float3[xSize * zSize];
        for (int i = 0, z = 0; z <= zSize; z++)
        {
            for (int x = 0; x <= xSize; x++)
            {
                float y = (Mathf.PerlinNoise(x * .015f * randomNum2, z * .02f * randomNum1) * 45f) + (Mathf.PerlinNoise(x * .04f * randomNum1, z * .025f * randomNum2) * 40f) ;
                vertices[i] = new float3(5*x + 1300, 5*y - 200, 5*z + 200);
                i++;
            }
        }

        triangles = new int[xSize * zSize * 6];
        int vert = 0;
        int tris = 0;
        for (int z = 0; z < zSize; z++)
        {
            for (int x = 0; x < xSize; x++)
            {
                triangles[tris + 0] = vert + 0;
                triangles[tris + 1] = vert + xSize + 1;
                triangles[tris + 2] = vert + 1;
                triangles[tris + 3] = vert + 1;
                triangles[tris + 4] = vert + xSize + 1;
                triangles[tris + 5] = vert + xSize + 2;

                vert++;
                tris += 6;

            };
            vert++;
        }
    }

    void ReadFile()
    {
        string PointsString = vertexFile.ToString();
        string[] arrayOfLines = PointsString.Split('\n');
        int index = 0;
        string[] values;
        vertices = new float3[xSizer * zSizer];
        float3 VectorNew;
        while (index < arrayOfLines.Length -1)
        {
            values = arrayOfLines[index].Split(' ');
            VectorNew = new float3(((float.Parse(values[1], CultureInfo.InvariantCulture)- zCorrection) / 10),
                               ((float.Parse(values[2], CultureInfo.InvariantCulture)) / 6),
                               ((float.Parse(values[0], CultureInfo.InvariantCulture)- xCorrection) / 10));
            vertices[index] = VectorNew;
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
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            if (vertices[i].y < 210)
            {
                colors[i] = color2;
            }
            else { colors[i] = color3; }
        }

    }

    public Vector3[] float3ToVector3Array(float3[] points) {
        Vector3[] list = new Vector3[points.Length];
        int i = 0;
        foreach(float3 point in points)
        {
            list[i] = new Vector3(point.x, point.y, point.z);
            i++;
        }
        return list;
    }

    float3[] Vector3Tofloat3Array(Vector3[] points)
    {
        float3[] list = new float3[points.Length];
        int i = 0;
        foreach (Vector3 point in points)
        {
            list[i] = new float3(point.x, point.y, point.z);
            i++;
        }
        return list;
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
        Debug.Log(" Verts: " + vertices.Length + " trians: " + triangles.Length);
        mesh.Clear();
        mesh.vertices = float3ToVector3Array(vertices);
        mesh.triangles = triangles;
        mesh.colors = colors;
        mesh.RecalculateNormals();
        normals = Vector3Tofloat3Array(mesh.normals);
    }
}
