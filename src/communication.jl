counter = 0

function start!(;host = "127.0.0.1", port = 2000)
    #start server
    println("###############################################")
    println("###############SERVER OPENED###################")
    println("###############################################")

    ## PERSISTENT LOOP
    WebSockets.listen!(host, port) do ws
        # FOR EACH MESSAGE SENT FROM CLIENT
        
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
            return
        end


        # ANALYSIS
        #try
            # DESERIALIZE MESSAGE
            problem = JSON.parse(msg)

            # MAIN ALGORITHM
            println("READING DATA")

            # CONVERT MESSAGE TO RECEIVER TYPE
            receiver = Receiver(problem)

            # SOLVE
            if counter == 0
                println("First run will take a while.")
                println("Julia needs to compile the code for the first run.")
            end
            
            # OPTIMIZATION
            @time FDMoptim!(receiver, ws)
           
        #catch error
            #println("INVALID INPUT")
            #println("CHECK PARAMETER BOUNDS")
            #println(error)        
        #end
        
        println("DONE")
        global counter += 1
        println("Counter $counter")
end
