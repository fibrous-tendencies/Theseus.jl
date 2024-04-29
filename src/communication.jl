cancel = false
simulating = false
counter = 0

function FDMsolve!(;host = "127.0.0.1", port = 2000)
    #start server
    println("###############################################")
    println("###############SERVER OPENED###################")
    println("###############################################")

    ## PERSISTENT LOOP
    WebSockets.listen!(host, port) do ws
        # FOR EACH MESSAGE SENT FROM CLIENT
        """
        for msg in ws
            try
            @async readMSG(msg, ws)
            catch error
                println(error)
            end
        """

        for msg in ws
            
             readMSG(msg, ws)

            
        end
    end
end

function readMSG(msg, ws)
        # ACKNOWLEDGE
        println("MSG RECEIVED")

        # FIRST MESSAGE
        if msg == "init"
            println("CONNECTION INITIALIZED")
            return
        end

        if msg == "cancel"
            println("Operation Cancelled")
            global cancel = true
            return
        end

        if simulating == true
            println("Simulation in progress")
            return
        end

        # ANALYSIS
        try
            # DESERIALIZE MESSAGE
            @time problem = JSON.parse(msg)

            # MAIN ALGORITHM
            println("READING DATA")

            # CONVERT MESSAGE TO RECEIVER TYPE
            @time receiver = Receiver(problem)

            # SOLVE
            if counter == 0
                println("First run will take a while! :--)")
                println("Julia needs to compile the code for the first run.")
            end
            
            # OPTIMIZATION
            global simulating = true
            @time FDMoptim!(receiver, ws)
           
        catch error
            println("INVALID INPUT")
            println("CHECK PARAMETER BOUNDS")
           println(error)
        end
        
        println("DONE")
        global simulating = false
        global counter += 1
        println("Counter $counter")
end
