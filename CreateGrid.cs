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
            InstantiateGrid(vertices);
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

    public void InstantiateGrid(List<Vector3> verts)
    {
        grid = new Grid(verts);
    }

    public void WriteString()
    {
        string path = "Assets/Output/test.txt";
        StreamWriter writer = new StreamWriter(path, false);
        foreach (Cell cell in grid.cells)
        {
            writer.WriteLine(cell.x);
        }
        writer.Close();
    }
}

public class Grid : Component
{
    public List<Vector3> OriginalPC;
    public List<Cell> cells = new List<Cell>();

    public Grid(List<Vector3> originalPC)
    {
        OriginalPC = originalPC;
        foreach (Vector3 point in originalPC)
        {
            cells.Add(new Cell(point));
        }
    }
}

public class Cell : Component
{
    public float x;
    public float y;
    public float z;

    public Cell(Vector3 loc)
    {
        x = loc.x;
        y = loc.y;
        z = loc.z;
    }
}

