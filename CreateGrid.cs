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
        setRunoffScores(grid);
        foreach (Cell cell in grid.cells)
        {
            if (cell.y == 0)
            {
                cell.curvature = 0;
                cell.relativeHeight1 = 0;
                cell.relativeHeight2 = 0;
                cell.relativeHeight3 = 0;
                cell.relativeSlope = 0;
                cell.relativeAspect = 0;
                cell.dRM1 = 0;
                cell.averageRunoff1 = 0;
                cell.averageRunoff2 = 0;
                cell.averageRunoff3 = 0;

            }
            else
            {
                cell.curvature = computeESRICurvature(cell, 2, 4);
                cell.relativeHeight1 = relativeHeight(cell.index, grid, 1);
                cell.relativeHeight2 = relativeHeight(cell.index, grid, 2);
                cell.relativeHeight3 = relativeHeight(cell.index, grid, 4);
                cell.relativeSlope = relativeSlope(cell.index, grid, 1);
                cell.relativeAspect = relativeAspect(cell.index, grid, 1);
                cell.dRM1 = DistTo(cell.x, cell.z, Correct2D(RM1, xCorrection, zCorrection));
                cell.averageRunoff1 = averageRunoff(grid, cell, 2);
                cell.averageRunoff2 = averageRunoff(grid, cell, 4);
                cell.averageRunoff3 = averageRunoff(grid, cell, 6);

            }
            //cell.dLN1 = Mathf.Pow(HandleUtility.DistancePointLine(new float3(cell.x, cell.y, cell.z), vertices[10], vertices[150800]), 2);
        }
        getDistanceToLines(grid);
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
                + " " + cell.relativeHeight1 + " " + cell.relativeSlope + " " + cell.relativeAspect
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
        int xLoc = getXFromIndex(index);
        int zLoc = getZFromIndex(index);

        if (xLoc < 0 + dist || xLoc > xSize - dist || zLoc < 0 + dist || zLoc > zSize - dist)
        {
            return 0;
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
            colors[i] = new Color(1f * (grid.cells[i].relativeHeight1 / 5), 1f * (grid.cells[i].relativeHeight1 / 5), 1f * (1 - ((grid.cells[i].relativeHeight1) / 5)), 1f);
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
    public void setMeshdLN1Colors()
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(1f * (20/grid.cells[i].distToSkeleton), 1f * (20/grid.cells[i].distToSkeleton), 1f * (20/grid.cells[i].distToSkeleton), 1f);
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
        int[] result = getRunoffPatterns(grid, array, 300, 20f);
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

    public void setMeshAverageRunoffColors(Grid grid)
    {
        colors = new Color[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            colors[i] = new Color(0f, 1f * (grid.cells[i].averageRunoff1 / 10), 1f * (grid.cells[i].averageRunoff1 / 10), 1f);
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

    public float averageRunoff(Grid grid, Cell cell, int dist)
    {
        int runoffSum = 0;
        int xLoc = getXFromIndex(cell.index);
        int zLoc = getZFromIndex(cell.index);
        float numOfCells = Mathf.Pow((1 + (2*dist)), 2);
        if(xLoc < 0 + dist || xLoc > xSize - dist || zLoc < 0 + dist || zLoc > zSize - dist)
        {
            return 0;
        }
        for( int i = -dist; i < dist; i++)
        {
            for (int j = -dist; j < dist; j++)
            {
                try
                {
                    runoffSum += grid.cells[getIndexFromLoc(xLoc + i, zLoc + j)].runoffScore;
                }
                catch { 
                    
                    Debug.Log(" index error x" + (xLoc + i) + " z " + (zLoc + j)); }
            }
        }
        return runoffSum / numOfCells;
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

    public void getDistanceToLines(Grid grid)
    {
        List<Vector3> list = new List<Vector3>
        {
           new Vector3(2003, 0, 937),
           new Vector3(1915, 0, 932),
           new Vector3(1885, 0, 909),
           new Vector3(1877, 0, 849),
           new Vector3(1717, 0, 721),
           new Vector3(1650, 0, 655),
           new Vector3(1571, 0, 644),
        };
        List<Vector3> list2 = new List<Vector3>
        {
           new Vector3(1712, 0, 776),
           new Vector3(1707, 0, 745),
           new Vector3(1653, 0, 708),
           new Vector3(1627, 0, 716),
           new Vector3(1627, 0, 690),
           new Vector3(1595, 0, 674),
           new Vector3(1579, 0, 651),
        };
        List<Vector3> list3 = new List<Vector3>
        {
           new Vector3(1820, 0, 602),
           new Vector3(1730, 0, 633),
           new Vector3(1695, 0, 664),
           new Vector3(1669, 0, 664),
        };
        List<Vector3> list4 = new List<Vector3>
        {
           new Vector3(1572, 0, 642),
           new Vector3(1539, 0, 607),
           new Vector3(1516, 0, 636),
           new Vector3(1502, 0, 641),
           new Vector3(1488, 0, 632),
           new Vector3(1491, 0, 592),
           new Vector3(1451, 0, 592),
           new Vector3(1442, 0, 627),
           new Vector3(1426, 0, 652),
           new Vector3(1407, 0, 620),
           new Vector3(1407, 0, 566),
           new Vector3(1378, 0, 552),
           new Vector3(1342, 0, 555),
           new Vector3(1297, 0, 580),
           new Vector3(1248, 0, 566),
           new Vector3(1179, 0, 557),
           new Vector3(1148, 0, 617),
           new Vector3(1125, 0, 617),
           new Vector3(1098, 0, 635),
           new Vector3(1022, 0, 556),
           new Vector3(995, 0, 562),
           new Vector3(942, 0, 616),
           new Vector3(845, 0, 581),
           new Vector3(794, 0, 601),
           new Vector3(788, 0, 570),
           new Vector3(809, 0, 554),
           new Vector3(784, 0, 550),
           new Vector3(747, 0, 579),
           new Vector3(653, 0, 597),
           new Vector3(558, 0, 582),
           new Vector3(528, 0, 552),
           new Vector3(576, 0, 499),
           new Vector3(576, 0, 466),
           new Vector3(545, 0, 407),
           new Vector3(558, 0, 383),
           new Vector3(608, 0, 420),
           new Vector3(620, 0, 409),
           new Vector3(618, 0, 390),
           new Vector3(633, 0, 337),
           new Vector3(627, 0, 326),
           new Vector3(582, 0, 345),
           new Vector3(563, 0, 334),
           new Vector3(571, 0, 319),
           new Vector3(613, 0, 292),
           new Vector3(635, 0, 271),
           new Vector3(534, 0, 232),
           new Vector3(550, 0, 275),
           new Vector3(522, 0, 294),
           new Vector3(422, 0, 279),
        };
        List<Vector3> list5 = new List<Vector3>
        {
           new Vector3(684, 0, 975),
           new Vector3(723, 0, 896),
           new Vector3(704, 0, 858),
           new Vector3(656, 0, 824),
           new Vector3(668, 0, 744),
           new Vector3(721, 0, 674),
           new Vector3(739, 0, 627),
           new Vector3(762, 0, 603),
           new Vector3(745, 0, 565),
        };
        List<Vector3> list6 = new List<Vector3>
        {
           new Vector3(906, 0, 319),
           new Vector3(945, 0, 414),
           new Vector3(1049, 0, 541),
           new Vector3(1029, 0, 561),
        };


        List<List<Vector3>> allLists = new List<List<Vector3>>();
        allLists.Add(list);
        allLists.Add(list2);
        allLists.Add(list3);
        allLists.Add(list4);
        allLists.Add(list5);
        allLists.Add(list6);

        foreach (Cell cell in grid.cells)
        {
            float smallestDist = 9999f;
            float currentDist = 9999f;
            if (cell.y != 0)
            {
                foreach (List<Vector3> sublist in allLists)
                {

                    for (int index = 0; index < sublist.Count - 1; index++)
                    {
                        currentDist = HandleUtility.DistancePointLine(new Vector3(cell.x, 0, cell.z), sublist[index], sublist[index + 1]);
                        if (currentDist < smallestDist)
                        {
                            smallestDist = currentDist;
                        }
                    }
                }
                    cell.distToSkeleton = smallestDist;
                }
            else
                {
                    cell.distToSkeleton = 9999999999f;

                }
            
        }

    }

    float distanceToLine(Vector3 point, Vector3 start, Vector3 end)
    {
        float distance = Mathf.Abs(
            ((end.x - start.x) * (start.z - point.z)) - ((start.x - point.x) * (end.z - start.z))) /
            Mathf.Pow((Mathf.Pow(end.x - start.x, 2) + Mathf.Pow(end.z - start.z, 2)), 0.5f);
        return distance;
    }


    void WriteAll()
    {
        Mesh mesh1983;
        Mesh mesh1997;
        Mesh mesh2008;
        Mesh mesh2012;
        Mesh mesh2018;
        Grid grid1983;
        Grid grid1997;
        Grid grid2008;
        Grid grid2012;
        Grid grid2018;


        //1983
        meshGenerator.StartPipe(meshGenerator.vertexFile1983);
        mesh1983 = meshGenerator.mesh;
        InstantiateGrid(mesh1983);
        grid1983 = grid;
        setRunoffScores(grid1983);
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

        List<Cell> cellsLowDiff = new List<Cell>();
        foreach (Cell cell in grid2008.cells) {

            float diff = (grid2012.cells[cell.index].y - cell.y);
            float diff2 = (cell.y - grid1997.cells[cell.index].y);
            float diff3 = (grid1997.cells[cell.index].y - grid1983.cells[cell.index].y);

            float maxDiff = 0.7f;
            if (Mathf.Pow(Mathf.Pow(diff,0.5f),2) < maxDiff && cell.y != 0 && Mathf.Pow(Mathf.Pow(diff2, 0.5f), 2) < maxDiff && Mathf.Pow(Mathf.Pow(diff3, 0.5f), 2) < maxDiff)
                {
                cellsLowDiff.Add(cell);
                Instantiate(dotgreen, cell.position, transform.rotation);
            }
        }

        int count = 0;
        float sum = 0f;
        float correction2018 = 0f;
        foreach (Cell cell in cellsLowDiff)
        {
            count++;
            sum += (grid2012.cells[cell.index].y - grid2018.cells[cell.index].y);
        }
        correction2018 = sum / count;

        foreach (Cell cell in cellsLowDiff)
        {
            Debug.Log("diff1 " + (grid2012.cells[cell.index].y - grid2018.cells[cell.index].y));
        }
        Debug.Log(" Mean: " + mean);
        string path = "Assets/Output/outputGridFull.txt";
        StreamWriter writer = new StreamWriter(path, false);
        writer.WriteLine("year interval x y hprevious hdifference hrelative1 hrelative2 hrelative3 slope aspect curvature dist averageRunoff1 averageRunoff2 averageRunoff3 discharge");
       /* //1983-1997
        foreach (Cell cell in grid1983.cells)
        {
            if (cell.y == 0 || double.IsNaN(cell.aspect)) { continue; }
            writer.WriteLine("1997 14 " + cell.x + " " + cell.z + " " + cell.y + " " + (grid1997.cells[cell.index].y - cell.y) + " " + cell.slope + " " + cell.aspect + " " + cell.curvature + " " + cell.runoffScore);
        } */
        //1997-2008
        foreach (Cell cell in grid1997.cells)
        {
            if (cell.y == 0 || double.IsNaN(cell.aspect)) { continue; }
            writer.WriteLine("2008 11 " + cell.x + " " + cell.z + " " + cell.y + " " + (grid2008.cells[cell.index].y - cell.y) + " "+ cell.relativeHeight1 + " " + cell.relativeHeight2 + " " + cell.relativeHeight3 + " " + cell.slope + " " + cell.aspect + " " + cell.curvature + " "  + cell.distToSkeleton + " " + cell.averageRunoff1 + " " + cell.averageRunoff2 + " " + cell.averageRunoff3 + " 73.9");
        }
        //2008-2012
        foreach (Cell cell in grid2008.cells)
        {
            if (cell.y == 0 || double.IsNaN(cell.aspect)) { continue; }
            writer.WriteLine("2012 4 " + cell.x + " " + cell.z + " " + cell.y + " " + (grid2012.cells[cell.index].y - cell.y) + " " + cell.relativeHeight1 + " " + cell.relativeHeight2 + " " + cell.relativeHeight3 + " " + cell.slope + " " + cell.aspect + " " + cell.curvature + " " +  cell.distToSkeleton + " " + cell.averageRunoff1 + " " + cell.averageRunoff2 + " " + cell.averageRunoff3 + " 95.5" );
        }
        //2012-2018
        foreach (Cell cell in grid2012.cells)
        {
            if (cell.y == 0 || double.IsNaN(cell.aspect)) { continue; }
            writer.WriteLine("2018 6 " + cell.x + " " + cell.z + " " + cell.y + " " + ((grid2018.cells[cell.index].y + correction2018) - cell.y) + " " + cell.relativeHeight1 + " " + cell.relativeHeight2 + " " + cell.relativeHeight3 + " " + cell.slope + " " + cell.aspect + " " + cell.curvature + " " + cell.distToSkeleton + " " + cell.averageRunoff1 + " " + cell.averageRunoff2 + " " + cell.averageRunoff3 + " 58.2" );
        }

        writer.Close();
    }
}



