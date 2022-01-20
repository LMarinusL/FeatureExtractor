using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class CreateGrid : MonoBehaviour
{
    public List<Vector3> vertices;
    public List<Vector3> normals;
    public MATList MATlist;
    public List<MATBall> MATcol;
    public Grid grid;

    void Update()
    {
        if (Input.GetKey(KeyCode.P))
        {
            getData();
            InstantiateGrid(vertices, normals);
            WriteString();
        }
    }

    void getData()
    {
        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
        GameObject MAT = GameObject.Find("MATLoader");
        ShrinkingBallSeg MATalg = MAT.GetComponent<ShrinkingBallSeg>();

        vertices = MATalg.vertices;
        normals = MATalg.normals;
        MATlist = MATalg.list;
        MATcol = MATlist.NewMATList;
    }

    public void InstantiateGrid(List<Vector3> verts, List<Vector3> normals)
    {
        grid = new Grid(verts, normals);
    }

    public void WriteString()
    {
        string path = "Assets/Output/outputGrid.txt";
        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine("x y h slope aspect");
        foreach (Cell cell in grid.cells)
        {
            writer.WriteLine(cell.x + " "+ cell.z + " " + cell.y + " " + 
                cell.slope + " " + cell.aspect);
        }
        writer.Close();
    }

}

public class Grid : Component
{ // list of grid cells in same grid order as input cells
    public List<Vector3> OriginalPC;
    public List<Cell> cells = new List<Cell>();

    public Grid(List<Vector3> originalPC, List<Vector3> originalNormals)
    {
        OriginalPC = originalPC;
        for (int i = 0; i < originalPC.ToArray().Length; i++)
        {
            cells.Add(new Cell(originalPC[i], originalNormals[i]));
        }
    }
}

public class Cell : Component
{
    public float x;
    public float y;
    public float z;
    public float slope;
    public float aspect;

    public Cell(Vector3 loc, Vector3 normal)
    {
        x = loc.x;
        y = loc.y;
        z = loc.z;
        slope = computeSlope(normal);
        aspect = computeAspect(normal);
    }

    float computeSlope(Vector3 normal)
    {
        float slope = Mathf.Tan(Mathf.Pow((Mathf.Pow(normal.x, 2f) + Mathf.Pow(normal.z, 2f)), 0.5f)/normal.y);
        return slope;
    }

    float computeAspect(Vector3 normal)
    {
        float aspect;
        if(normal.x > 0)
        {
            aspect = 90f - 57.3f*(Mathf.Atan(normal.z / normal.x));
        }
        else
        {
            aspect = 270f - 57.3f*(Mathf.Atan(normal.z / normal.x));
        }
        return aspect;
    }
}



