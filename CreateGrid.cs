using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System.Linq;
using System;
using Unity.Mathematics;

public class CreateGrid : MonoBehaviour
{
    public float3[] vertices;
    public float3[] normals;
    public MATList MATlist;
    public MATBall[] MATcol;
    public Grid grid;
    public Vector2 RM1 = new Vector2(659492f, 1020360f);
    public Vector2 RM2 = new Vector2(654296f, 1023740f);
    public Vector2 RM3 = new Vector2(658537f, 1032590f);
    float xCorrection;
    float zCorrection;
    public int xSize;
    public int zSize;
    public GameObject dotgreen;
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
        if (Input.GetKey(KeyCode.Alpha7))
        {
            setMeshContourColors();
        }
        if (Input.GetKeyDown(KeyCode.T))
        {
            int index = 0;
            int[] array = new int[vertices.Length];
            while ( index < vertices.Length)
            {
                array[index] = index;
                index++;
            }
            setMeshRunoffColors(array, 3000, 20f);
        }
        if (Input.GetKeyDown(KeyCode.Y))
        {
            List<int> startAt = new List<int>();
            for (int i = 0; i < 1000; i++)
            {
                startAt.Add(UnityEngine.Random.Range(100, 250000));
            }
            setMeshRunoffColors(startAt.ToArray(), 3000, 20f);
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            StartCoroutine(iterate(1000));
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

    public void InstantiateGrid(float3[] verts, float3[] normals)
    {
        grid = new Grid(verts, normals);
        foreach (Cell cell in grid.cells)
        {
            cell.relativeHeight = relativeHeight(cell.index, grid, 1);
            cell.relativeSlope = relativeSlope(cell.index, grid, 1);
            cell.relativeAspect = relativeAspect(cell.index, grid, 1);
            cell.dRM1 = DistTo(cell.x, cell.z, Correct2D(RM1, xCorrection, zCorrection));
            cell.dLN1 = Mathf.Pow(HandleUtility.DistancePointLine(new float3(cell.x, cell.y, cell.z), vertices[10], vertices[150800]), 2);
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
                + " " + HandleUtility.DistancePointLine(new float3(cell.x, cell.y, cell.z), vertices[10], vertices[400])
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

    public List<int> getIndicesOfSurroundingCells(int index, Grid grid, int dist)
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
        return (xLoc ) + (zLoc * xSize); 
    }
    
    public int getZFromIndex(int index)
    {
        int result = Mathf.FloorToInt(index / xSize);
        return result;
    }

    public int getXFromIndex(int index)
    {
        return index - (getZFromIndex(index) * xSize);
    }

    // COLORS
    // todo: get max values to set colors
    void setMeshSlopeColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f , 1f * (1 - grid.cells[i].slope/1.52f), 0f, 1f);
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
    void setMeshContourColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            if (grid.cells[i].y < 100)
            {
                colors[i] = new Color(1f, 1f , 1f, 1f);
            }
            else {
                colors[i] = new Color(0f, 0f, 0f , 1f);
            }
        }
        mesh.colors = colors;
    }
    void setMeshRunoffColors(int[] starts, int num, float margin)
    {
        int[] patterns = getRunoffPatterns(starts, num, margin);
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(0f , 1f * (grid.cells[i].runoffScore / 10), 1f * (grid.cells[i].runoffScore / 10), 1f);
        }
        mesh.colors = colors;
    }

    int[] getRunoffPatterns(int[] startingPoints, int numOfIterations, float margin)
    {
        List<int> patterns = new List<int>();
        List<int> currentPattern = new List<int>();
        foreach (int start in startingPoints)
        {
            patterns.Add(start);
            currentPattern.Clear();
            int previousIndex = 0;
            int ownIndex = start;
            bool keepRolling = true;
            int iteration = 0;
            while (keepRolling == true)
            {
                iteration++;
                
                if (iteration == numOfIterations) { keepRolling = false; }
                float ownHeight = grid.cells[ownIndex].y;
                List<int> possiblePaths = getIndicesOfSurroundingCells(ownIndex, grid, 1);
                int lowestHeightIndex = ownIndex;
                float lowestHeight = ownHeight + margin;
                foreach( int index in possiblePaths)
                { 
                    if (grid.cells[index].y < lowestHeight && index != previousIndex && grid.cells[index].y != 0 && currentPattern.Contains(index) == false)
                    {
                        lowestHeight = grid.cells[index].y;
                        lowestHeightIndex = index;
                    }
                }
                if (lowestHeightIndex == ownIndex) { keepRolling = false; }
                else {
                    grid.cells[lowestHeightIndex].runoffScore += 1;
                    patterns.Add(lowestHeightIndex);
                    currentPattern.Add(lowestHeightIndex);
                    previousIndex = ownIndex;
                    ownIndex = lowestHeightIndex;
                    }
            }
        }
        return patterns.ToArray();
    }

    IEnumerator iterate(int num)
    {
        List<int> startAt = new List<int>();
        for (int j = 0; j < num; j++)
        {
            startAt.Add(UnityEngine.Random.Range(100, 250000));
            setMeshRunoffColors(startAt.ToArray(), 3000, 20f);
            yield return new WaitForSeconds(.01f);
        }
    }
    /*
    void InstantiateRunoff(int[] starts, int num, float margin)
    {
        int[] patterns = getRunoffPatterns(starts, num, margin);
        foreach (int point in patterns)
        {
            GameObject dot = Instantiate(dotgreen, new Vector3(grid.cells[point].x, grid.cells[point].y, grid.cells[point].z),  transform.rotation);
            dot.GetComponent<MeshRenderer>().material.color = new Color((grid.cells[point].runoffScore / 10) * 1f, (grid.cells[point].runoffScore/10) *1f, (grid.cells[point].runoffScore / 10) * 1f, 1f);
            
        }
    }*/
}

public class Grid : Component
{ // list of grid cells in same grid order as input cells
    public float3[] OriginalPC;
    public Cell[] cells;

    public Grid(float3[] originalPC, float3[] originalNormals)
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
    public int runoffScore;



    public Cell(int i, float3 loc, float3 normal)
    {
        index = i;
        x = loc.x;
        y = loc.y;
        z = loc.z;
        slope = computeSlope(normal);
        aspect = computeAspect(normal);
        runoffScore = 0;
    }

    float computeSlope(float3 normal)
    {
        float slope = Mathf.Atan(Mathf.Pow((Mathf.Pow(normal.x, 2f) + Mathf.Pow(normal.z, 2f)), 0.5f)/normal.y);
        return slope;
    }

    float computeAspect(float3 normal)
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



