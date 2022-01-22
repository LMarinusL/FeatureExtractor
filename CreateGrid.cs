using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;

public class CreateGrid : MonoBehaviour
{
    public List<Vector3> vertices;
    public List<Vector3> normals;
    public MATList MATlist;
    public List<MATBall> MATcol;
    public Grid grid;
    public Vector2 RM1 = new Vector2(659492f, 1020360f);
    public Vector2 RM2 = new Vector2(654296f, 1023740f);
    public Vector2 RM3 = new Vector2(658537f, 1032590f);
    float xCorrection;
    float zCorrection;

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

        xCorrection = meshGenerator.xCorrection;
        zCorrection = meshGenerator.zCorrection;

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
        writer.WriteLine("x y h slope aspect RM1 RM2 RM3");
        foreach (Cell cell in grid.cells)
        {
            writer.WriteLine(cell.x + " "+ cell.z + " " + cell.y + " " + 
                cell.slope + " " + cell.aspect + " " + DistTo(cell.x, cell.z, Correct2D(RM1, xCorrection, zCorrection))
                + " " + DistTo(cell.x, cell.z, Correct2D(RM2, xCorrection, zCorrection))
                + " " + DistTo(cell.x, cell.z, Correct2D(RM3, xCorrection, zCorrection))
                + " " + HandleUtility.DistancePointLine(new Vector3(cell.x, cell.y, cell.z), vertices[10], vertices[400])
                );
        }
        writer.Close();
    }

    public float DistTo(float x, float y , Vector2 Point)
    {
        float dist = Mathf.Pow((Mathf.Pow(x - Point.x, 2f) + Mathf.Pow(y - Point.y, 2f)), 0.5f);
        return dist;
    }

    public Vector2 Correct2D(Vector2 point, float xcor, float ycor)
    {
        return new Vector2((point.x-xcor)/10 , (point.y - ycor) / 10);
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



