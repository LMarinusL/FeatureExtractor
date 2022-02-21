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
    public int[] triangles;
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
    public Mesh mesh;
    Color[] colors;
    MeshGenerator meshGenerator;

  void Start()
    {
        getData();
        InstantiateGrid(mesh);
        WriteString();
        Debug.Log("Output written");
    }
  

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            WriteAll();
        }
    }

    public void getData()
    {
        GameObject terrain = GameObject.Find("TerrainLoader");
        meshGenerator = terrain.GetComponent<MeshGenerator>();
        GameObject MAT = GameObject.Find("MATLoader");
        ShrinkingBallSeg MATalg = MAT.GetComponent<ShrinkingBallSeg>();
        //meshGenerator.StartPipe(meshGenerator.vertexFile2018);
        xCorrection = meshGenerator.xCorrection;
        zCorrection = meshGenerator.zCorrection;
        xSize = meshGenerator.xSizer;
        zSize = meshGenerator.zSizer;
        mesh = meshGenerator.mesh;

        vertices = meshGenerator.vertices;
        normals = meshGenerator.normals;
        triangles = meshGenerator.triangles;
        MATlist = MATalg.list;
        MATcol = MATlist.NewMATList;
    } 

    public void  InstantiateGrid(Mesh mesh)
    {
        grid = new Grid(meshGenerator.Vector3Tofloat3Array(mesh.vertices), 
            meshGenerator.Vector3Tofloat3Array(mesh.normals),
            mesh.triangles);
        foreach (Cell cell in grid.cells)
        {
            cell.curvature = computeESRICurvature(cell, 2, 4);
            cell.relativeHeight = relativeHeight(cell.index, grid, 1);
            cell.relativeSlope = relativeSlope(cell.index, grid, 1);
            cell.relativeAspect = relativeAspect(cell.index, grid, 1);
            cell.dRM1 = DistTo(cell.x, cell.z, Correct2D(RM1, xCorrection, zCorrection));
            //cell.dLN1 = Mathf.Pow(HandleUtility.DistancePointLine(new float3(cell.x, cell.y, cell.z), vertices[10], vertices[150800]), 2);
        }
        Instantiate(dotgreen, grid.cells[214228].position, transform.rotation);


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

    public float relativeHeight(int index, Grid grid, int dist)
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

    public float relativeSlope(int index, Grid grid, int dist)
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

    public float relativeAspect(int index, Grid grid, int dist)
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
    public void setMeshSlopeColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f , 1f * (1 - grid.cells[i].slope/1.52f), 0f, 1f);
        }
        mesh.colors = colors;
    }
    public void setMeshAspectColors()
    {
        colors = new Color[vertices.Length]; 
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].aspect/180), 1f * (grid.cells[i].aspect / 180), 1f * ((180 - grid.cells[i].aspect)/180), 1f);
        }
        mesh.colors = colors;
    }
    public void setMeshRelativeSlopeColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(Mathf.Pow(Mathf.Pow((grid.cells[i].relativeSlope*3), 2f), 0.5f),  Mathf.Pow(Mathf.Pow((grid.cells[i].relativeSlope*3), 2f), 0.5f), Mathf.Pow(Mathf.Pow((grid.cells[i].relativeSlope*3), 2f), 0.5f), 1f);
        }
        mesh.colors = colors;
    }
    public void setMeshRelativeAspectColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].relativeAspect / 50), 1f * (grid.cells[i].relativeAspect / 50), 1f * (1-((grid.cells[i].relativeAspect) / 50)), 1f);
        }
        mesh.colors = colors;
    }
    public void setMeshRelativeHeightColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].relativeHeight / 5), 1f * (grid.cells[i].relativeHeight / 5), 1f * (1 - ((grid.cells[i].relativeHeight) / 5)), 1f);
        }
        mesh.colors = colors;
    }
    public void setMeshCurveColors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (grid.cells[i].curvature), 1f * (grid.cells[i].curvature ), 1f * (grid.cells[i].curvature), 1f);

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

    void setRunoffScores(Grid grid)
    {
        int ind = 0;
        int[] array = new int[vertices.Length];
        while (ind < vertices.Length)
        {
            array[ind] = ind;
            ind++;
        }
        int[] result = getRunoffPatterns(grid, array, 3000, 20f);
    }

    public void setMeshRunoffColors(int[] starts, int num, float margin)
    {
        int[] patterns = getRunoffPatterns(grid, starts, num, margin);
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(0f , 1f * (grid.cells[i].runoffScore / 10), 1f * (grid.cells[i].runoffScore / 10), 1f);
        }
        mesh.colors = colors;
    }

    public IEnumerator iterate(int num)
    {
        List<int> startAt = new List<int>();
        for (int j = 0; j < num; j++)
        {
            startAt.Add(UnityEngine.Random.Range(100, 250000));
            setMeshRunoffColors(startAt.ToArray(), 3000, 20f);
            yield return new WaitForSeconds(.01f);
        }
    }

    int[] getRunoffPatterns(Grid grid, int[] startingPoints, int numOfIterations, float margin)
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

    void computeCurvatureY(Cell cell)
    {
        if (cell.y == 0 || cell.attachedTriangles.Count != 6) 
        {
            cell.curvatureY = 0f;
            return;
        } 
        else 
        {
                Face[] crossingFaces = new Face[2];
                Vector3[] crossingVertices = new Vector3[2];

                int indexNewFace = 0;
                foreach (Triangle triangle in cell.attachedTriangles)
                {
                    foreach (Face face in triangle.faces)
                    {
                        if (((face.startVertex.y < cell.y && face.endVertex.y > cell.y) ||
                            (face.startVertex.y > cell.y && face.endVertex.y < cell.y)) && indexNewFace< 2)
                        {
                            crossingFaces[indexNewFace] = face;
                            indexNewFace++;
                        }
                    }
                }
                int indexNewLoc = 0;
                if(indexNewFace != 2)
                {
                    cell.curvatureY = 2.5f;
                    return;
                }
                foreach (Face face in crossingFaces)
                {
                    if (face.startVertex.y != 0 && face.endVertex.y != 0)
                    {
                        float xStep = face.startVertex.x - face.endVertex.x;
                        float zStep = face.startVertex.z - face.endVertex.z;

                        float ratio = ((face.startVertex.y - cell.y) / (face.endVertex.y - face.startVertex.y));
                        Vector3 crossPoint = new Vector3(face.startVertex.x + (xStep * ratio), cell.y, face.startVertex.z + (zStep * ratio));
                        crossingVertices[indexNewLoc] = crossPoint;
                        indexNewLoc++;
                    }
                    else
                    {
                        cell.curvatureY = 0;
                        return;
                    }
                }
                Vector3 expectedLocation = new Vector3(((crossingVertices[0].x - crossingVertices[1].x) / 2) + crossingVertices[1].x,
                                                        ((crossingVertices[0].y - crossingVertices[1].y) / 2) + crossingVertices[1].y,
                                                        ((crossingVertices[0].z - crossingVertices[1].z) / 2) + crossingVertices[1].z);
                cell.curvatureY = Mathf.Pow(Mathf.Pow(Vector3.Distance(cell.position, expectedLocation), 2), 0.5f);
            }
    }


    float computeESRICurvature(Cell cell, int dist, int connectivity)
    {
        //http://help.arcgis.com/en/arcgisdesktop/10.0/help/index.html#//00q90000000t000000
        //D = [(Z4 + Z6) /2 - Z5] / L2
        //E = [(Z2 + Z8) /2 - Z5] / L2
        //The output of the Curvature tool is the second derivative of the surface—for example, the slope of the slope—such that:
        //Curvature = -2(D + E) * 100

        if (cell.y == 0 || cell.attachedTriangles.Count != 6)
        { 
            return 0f;
        }
        else
        {
            Cell Z2 = null;
            Cell Z4 = null;
            Cell Z6 = null;
            Cell Z8 = null;
            List<Cell> cells = getSurroundingCells(cell, grid, dist, connectivity);
            if (cells.Count == connectivity)
            {
                foreach (Cell surroundingCell in cells)
                {
                    if (surroundingCell.x == cell.x && surroundingCell.z < cell.z)
                    {
                        Z8 = surroundingCell;
                    }
                    if (surroundingCell.x == cell.x && surroundingCell.z > cell.z)
                    {
                        Z2 = surroundingCell;
                    }
                    if (surroundingCell.x > cell.x && surroundingCell.z == cell.z)
                    {
                        Z6 = surroundingCell;
                    }
                    if (surroundingCell.x < cell.x && surroundingCell.z == cell.z)
                    {
                        Z4 = surroundingCell;
                    }
                }
            }
            else 
            { 
            return 0f; 
            }
            
            float xStep = cell.attachedFaces[2].startVertex.x - cell.attachedFaces[2].endVertex.x;
            float zStep = cell.attachedFaces[0].startVertex.z - cell.attachedFaces[0].endVertex.z;
            float D = (((Z4.y + Z6.y) / 2) - cell.y) / (zStep * 2);
            float E = (((Z2.y + Z8.y) / 2) - cell.y) / (xStep * 2);
            float curvature = -2 * (D + E);
            return curvature;
        }
    }


    public List<Cell> getSurroundingCells(Cell own, Grid grid, int dist, int connectivity)
    {
        int xLoc = getXFromIndex(own.index);
        int zLoc = getZFromIndex(own.index);
        List<Cell> cells = new List<Cell>();
        if (connectivity != 8 && connectivity != 4)
        {
            Debug.Log(" incorrect connectivity ");
            return cells;
        }
        if (xLoc > 0 + dist)
        {
            cells.Add(grid.cells[getIndexFromLoc(xLoc - dist, zLoc)]);
            if (zLoc > 0 + dist && connectivity == 8)
            {
                cells.Add(grid.cells[getIndexFromLoc(xLoc - dist, zLoc - dist)]);
            }
            if (zLoc < (zSize - dist) && connectivity == 8)
            {
                cells.Add(grid.cells[getIndexFromLoc(xLoc - dist, zLoc + dist)]);
            }
        }
        if (zLoc > 0 + dist)
        {
            cells.Add(grid.cells[getIndexFromLoc(xLoc, zLoc - dist)]);
        }
        if (zLoc < (zSize - dist))
        {
            cells.Add(grid.cells[getIndexFromLoc(xLoc, zLoc + dist)]);
        }
        if (xLoc < (xSize - dist))
        {
            cells.Add(grid.cells[getIndexFromLoc(xLoc + dist, zLoc)]);
            if (zLoc > 0 + dist && connectivity == 8)
            {
                cells.Add(grid.cells[getIndexFromLoc(xLoc + dist, zLoc - dist)]);
            }
            if (zLoc < (zSize - dist) && connectivity == 8)
            {
                cells.Add(grid.cells[getIndexFromLoc(xLoc + dist, zLoc + dist)]);
            }
        }
        return cells;
    }


    void WriteAll()
    {
        Mesh mesh1997;
        Mesh mesh2008;
        Mesh mesh2012;
        Mesh mesh2018;
        Grid grid1997;
        Grid grid2008;
        Grid grid2012;
        Grid grid2018;


        //1997
        meshGenerator.StartPipe(meshGenerator.vertexFile1997);
        mesh1997 = meshGenerator.mesh;
        InstantiateGrid(mesh1997);
        grid1997 = grid;
        setRunoffScores(grid1997);
        //2008
        meshGenerator.StartPipe(meshGenerator.vertexFile2008);
        mesh2008 = meshGenerator.mesh;
        InstantiateGrid(mesh2008);
        grid2008 = grid;
        setRunoffScores(grid2008);
        //2012
        meshGenerator.StartPipe(meshGenerator.vertexFile2012);
        mesh2012 = meshGenerator.mesh;
        InstantiateGrid(mesh2012);
        grid2012 = grid;
        setRunoffScores(grid2012);

        //2018
        meshGenerator.StartPipe(meshGenerator.vertexFile2018);
        mesh2018 = meshGenerator.mesh;
        InstantiateGrid(mesh2018);
        grid2018 = grid;
        setRunoffScores(grid2018);





        string path = "Assets/Output/outputGridFull.txt";
        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine("year interval x y hprevious hcurrent hdifference relativeHeight slope relativeSlope aspect relativeAspect");
        //1997-2008
        foreach (Cell cell in grid1997.cells)
        {
            if (cell.y == 0) { continue; }
            writer.WriteLine(" 2008 11 " + cell.curvature + " " + cell.z + " " + cell.y + " " +
                cell.slope + " " + cell.aspect + " ");
        }
        //2008-2012
        foreach (Cell cell in grid2008.cells)
        {
            if (cell.y == 0) { continue; }
            writer.WriteLine(" 20012 4 " + cell.curvature + " " + cell.z + " " + cell.y + " " +
                cell.slope + " " + cell.aspect + " ");
        }
        //2012-2018
        foreach (Cell cell in grid2012.cells)
        {
            if (cell.y == 0) { continue; }
            writer.WriteLine(" 2018 6 " + cell.curvature + " " + cell.z + " " + cell.y + " " +
                cell.slope + " " + cell.aspect + " ");
        }

        writer.Close();
    }
}



