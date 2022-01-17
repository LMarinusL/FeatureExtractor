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

    void Update()
    {
        if (Input.GetKey(KeyCode.P))
        {
            getData();
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

    public void WriteString()
    {
        string path = "Assets/Output/test.txt";
        StreamWriter writer = new StreamWriter(path, false);
        foreach (MATBall ball in MATcol)
        {
            writer.WriteLine(ball.Score);
        }
        writer.Close();
    }
}

public class GridComponent : Component
{
    public List<Vector3> OriginalPC;

    public GridComponent(List<Vector3> originalPC)
    {
        OriginalPC = originalPC;
    }
}

public class CellComponent : Component
{
    public Vector3 Loc;

    public CellComponent(Vector3 loc)
    {
        Loc = loc;
    }
}

