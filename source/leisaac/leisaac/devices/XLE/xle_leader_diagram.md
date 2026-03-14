<!-- # XLE_leader Block Diagram

This diagram illustrates how `xle_leader.py` handles input from the keyboard and physical SO101 leader arms, performs coordinate transformations, and aggregates them for the Isaac Lab environment. -->

```mermaid
graph TD
    subgraph Input_Devices [Input Devices]
        KB[Keyboard]
        LL[Left SO101 Leader Arm]
        RL[Right SO101 Leader Arm]
    end

    subgraph XLE_Leader_Logic [XLE_leader Logic]
        KB_Map[Keyboard Event Mapping]
        Speed_Ctrl[Speed Level Control]
        Coord_Trans[Coordinate Transformation]
        State_Agg[State Aggregator]
    end

    subgraph Isaac_Sim [Isaac Sim]
        Env_Scene[Environment Scene]
    end

    KB -->|Key Press| KB_Map
    KB -->|Key 1,2,3| Speed_Ctrl
    Speed_Ctrl -->|Scaling| KB_Map
    KB_Map -->|Local Vel| Coord_Trans
    
    Env_Scene -->|Heading theta| Coord_Trans
    
    LL -->|Joint States| State_Agg
    RL -->|Joint States| State_Agg
    Coord_Trans -->|World Action| State_Agg
    
    State_Agg -->|Unified Dict| Output[Get Device State]
```
