using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;

public class CreateGrid : MonoBehaviour
{
    public Vector3[] vertices;
    public Vector3[] normals;
    public MATList MATlist;
    public MATBall[] MATcol;
    public Grid grid;
    public Vector2 RM1 = new Vector2(659492f, 1020360f);
    public Vector2 RM2 = new Vector2(654296f, 1023740f);
    public Vector2 RM3 = new Vector2(658537f, 1032590f);
    float xCorrection;
    float zCorrection;
    int xSize;
    int zSize;
    Mesh mesh;
    Color[] colors;




    void Update()
    {
        if (Input.GetKey(KeyCode.P))
        {
            getData();
            InstantiateGrid(vertices, normals);
            WriteString();
            Debug.Log("Output written");
        }
        if (Input.GetKey(KeyCode.Alpha1))
        {
            setMeshSlopeColors();
        }
        if (Input.GetKey(KeyCode.Alpha2))
        {
            setMeshAspectColors();
        }
        if (Input.GetKey(KeyCode.Alpha3))
        {
            setMeshRelativeSlopeColors();
        }
        if (Input.GetKey(KeyCode.Alpha4))
        {
            setMeshRelativeAspectColors();
        }
        if (Input.GetKey(KeyCode.Alpha5))
        {
            setMeshRelativeHeightColors();
        }
        if (Input.GetKey(KeyCode.Alpha6))
        {
            setMeshdLN1Colors();
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
        xSize = meshGenerator.xSizer;
        zSize = meshGenerator.zSizer;
        mesh = meshGenerator.mesh;

        vertices = meshGenerator.vertices;
        normals = meshGenerator.normals;
        MATlist = MATalg.list;
        MATcol = MATlist.NewMATList;
    }

    public void InstantiateGrid(Vector3[] verts, Vector3[] normals)
    {
        grid = new Grid(verts, normals);
        foreach (Cell cell in grid.cells)
        {
            cell.relativeHeight = relativeHeight(cell.index, grid, 1);
            cell.relativeSlope = relativeSlope(cell.index, grid, 1);
            cell.relativeAspect = relativeAspect(cell.index, grid, 1);
            cell.dRM1 = DistTo(cell.x, cell.z, Correct2D(RM1, xCorrection, zCorrection));
            cell.dLN1 = Mathf.Pow(HandleUtility.DistancePointLine(new Vector3(cell.x, cell.y, cell.z), vertices[10], vertices[150800]), 2);
        }
    }

    public void WriteString()
    {
        string path = "Assets/Output/outputGrid.txt";
        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine("x y h slope aspect RM1 RM2 RM3 relativeHeight relativeSlope relativeAspect");
        foreach (Cell cell in grid.cells)
        {
            if(cell.y == 0) { continue; }
            writer.WriteLine(cell.x + " "+ cell.z + " " + cell.y + " " + 
                cell.slope + " " + cell.aspect + " " 
                + DistTo(cell.x, cell.z, Correct2D(RM1, xCorrection, zCorrection))
                + " " + DistTo(cell.x, cell.z, Correct2D(RM2, xCorrection, zCorrection))
                + " " + DistTo(cell.x, cell.z, Correct2D(RM3, xCorrection, zCorrection))
                + " " + HandleUtility.DistancePointLine(new Vector3(cell.x, cell.y, cell.z), vertices[10], vertices[400])
                + " " + cell.relativeHeight + " " + cell.relativeSlope + " " + cell.relativeAspect
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

    List<int> getIndicesOfSurroundingCells(int index, Grid grid, int dist)
    {
        Cell own = grid.cells[index];
        int xLoc = getXFromIndex(index);
        int zLoc = getZFromIndex(index);
        List<int> indices = new List<int>();
        if(xLoc > 0 + dist )
        {
            indices.Add(getIndexFromLoc(xLoc - dist, zLoc));
            if (zLoc > 0 + dist)
            {
                indices.Add(getIndexFromLoc(xLoc - dist, zLoc- dist));
            }
            if (zLoc < (zSize - dist))
            {
                indices.Add(getIndexFromLoc(xLoc - dist, zLoc + dist));
            }
        }
        if (zLoc > 0 + dist)
        {
            indices.Add(getIndexFromLoc(xLoc , zLoc- dist));
        }
        if (zLoc < (zSize - dist))
        {
            indices.Add(getIndexFromLoc(xLoc, zLoc + dist));
        }
        if (xLoc < (xSize- dist))
        {
            indices.Add(getIndexFromLoc(xLoc + dist, zLoc));
            if (zLoc > 0 + dist)
            {
                indices.Add(getIndexFromLoc(xLoc + dist, zLoc - dist));
            }
            if (zLoc < (zSize - dist))
            {
                indices.Add(getIndexFromLoc(xLoc + dist, zLoc + dist));
            }
        }
        return indices;
    }

    float relativeHeight(int index, Grid grid, int dist)
    {
        List<int> indices = getIndicesOfSurroundingCells(index, grid, dist);
        float averageHeight = 0f;
        float heightSum = 0f;
        int numOfCells = 0;
        foreach (int i in indices)
        {
            if (grid.cells[i].y != 0) // only take vertices that are not at the height 0
            {
                heightSum = heightSum + grid.cells[i].y;
                numOfCells++;
            }
        }
        if (numOfCells == 0) // if there are no cells around it that are not at height zero, prevent dividing by zero
        {
            return 0f;
        }
        averageHeight = heightSum / numOfCells;
        float heightOwn = grid.cells[index].y;
        return averageHeight - heightOwn;
    }

    float relativeSlope(int index, Grid grid, int dist)
    {
        List<int> indices = getIndicesOfSurroundingCells(index, grid, dist);
        float averageSlope = 0f;
        float slopeSum = 0f;
        int numOfCells = 0;
        foreach (int i in indices)
        {
            if (grid.cells[i].y != 0) // only take vertices that are not at the height 0
            {
                slopeSum = slopeSum + grid.cells[i].slope;
                numOfCells++;
            }
        }
        if (numOfCells == 0)
        {
            return 0f;
        }
        averageSlope = slopeSum / numOfCells;
        float slopeOwn = grid.cells[index].slope;
        return averageSlope - slopeOwn;
    }

    float relativeAspect(int index, Grid grid, int dist)
    {
        List<int> indices = getIndicesOfSurroundingCells(index, grid, dist);
        float averageAspect = 0f;
        float aspectSum = 0f;
        int numOfCells = 0;
        foreach (int i in indices)
        {
            if (grid.cells[i].y != 0) // only take vertices that are not at the height 0
            {
                aspectSum = aspectSum + grid.cells[i].aspect;
                numOfCells++;
            }
        }
        if (numOfCells == 0)
        {
            return 0f;
        }
        averageAspect = aspectSum / numOfCells;
        float aspectOwn = grid.cells[index].aspect;
        return averageAspect - aspectOwn;
    }

    public int getIndexFromLoc(int xLoc, int zLoc)
    {
        return (zLoc ) + (xLoc * zSize);
    }
    
    public int getXFromIndex(int index)
    {
        int result = Mathf.FloorToInt(index / zSize);
        return result;
    }

    public int getZFromIndex(int index)
    {
        return index - (Mathf.FloorToInt(index / zSize)* zSize);
    }

    // COLORS
    // todo: get max values to set colors
    void setMeshSlopeColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].slope/1.52f), 1f * (grid.cells[i].slope/1.52f), 1f * (1 - (grid.cells[i].slope/1.52f)), 1f);
        }
        mesh.colors = colors;
    }
    void setMeshAspectColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].aspect/180), 1f * (grid.cells[i].aspect / 180), 1f * ((180 - grid.cells[i].aspect)/180), 1f);
        }
        mesh.colors = colors;
    }
    void setMeshRelativeSlopeColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(Mathf.Pow(Mathf.Pow((grid.cells[i].relativeSlope*3), 2f), 0.5f),  Mathf.Pow(Mathf.Pow((grid.cells[i].relativeSlope*3), 2f), 0.5f), Mathf.Pow(Mathf.Pow((grid.cells[i].relativeSlope*3), 2f), 0.5f), 1f);
        }
        mesh.colors = colors;
    }
    void setMeshRelativeAspectColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].relativeAspect / 50), 1f * (grid.cells[i].relativeAspect / 50), 1f * (1-((grid.cells[i].relativeAspect) / 50)), 1f);
        }
        mesh.colors = colors;
    }
    void setMeshRelativeHeightColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].relativeHeight / 20), 1f * (grid.cells[i].relativeHeight / 20), 1f * (1-((grid.cells[i].relativeHeight) /20)), 1f);
        }
        mesh.colors = colors;
    }
    void setMeshdLN1Colors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].dLN1/10000), 1f * (grid.cells[i].dLN1/10000), 1f * (grid.cells[i].dLN1/10000), 1f);
        }
        mesh.colors = colors;
    }

    void getRunoffPatterns(Vector3[] startingPoints)
    {
        // for each starting point
            // add the starting point to an array 
            // previousCell = index 0
            // ownindex = startingpoint
                // while the drop can continue rolling
                    // go to next lowest point that is not a previous point, add next point to the array
                    // previousindex = ownindex
                    // ownindex = nextcell
                    // getIdexOfSurroundingCells if not previouscell
                    // points cannot have height 0

    }

}

public class Grid : Component
{ // list of grid cells in same grid order as input cells
    public Vector3[] OriginalPC;
    public Cell[] cells;

    public Grid(Vector3[] originalPC, Vector3[] originalNormals)
    {
        OriginalPC = originalPC;
        cells = new Cell[originalPC.Length];
        for (int i = 0; i < originalPC.Length; i++)
        {
            cells[i] = new Cell(i, originalPC[i], originalNormals[i]);
        }
    }
}

public class Cell : Component
{
    public int index;
    public float x;
    public float y;
    public float z;
    public float slope;
    public float aspect;
    public float relativeHeight;
    public float relativeSlope;
    public float relativeAspect;
    public float dRM1;
    public float dLN1;



    public Cell(int i, Vector3 loc, Vector3 normal)
    {
        index = i;
        x = loc.x;
        y = loc.y;
        z = loc.z;
        slope = computeSlope(normal);
        aspect = computeAspect(normal);
    }

    float computeSlope(Vector3 normal)
    {
        float slope = Mathf.Atan(Mathf.Pow((Mathf.Pow(normal.x, 2f) + Mathf.Pow(normal.z, 2f)), 0.5f)/normal.y);
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



