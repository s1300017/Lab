// Ollamaカスタムmain.go: /api/versionだけGinのアクセスログを抑制
package main

import (
	"github.com/gin-gonic/gin"
	"net/http"
)

func main() {
	r := gin.New()

	// /api/versionだけロギング抑制
	r.Use(func(c *gin.Context) {
		if c.Request.URL.Path == "/api/version" {
			c.Next()
			return
		}
		gin.Logger()(c)
	})

	// 通常のOllamaサーバーのルーティング（本来はollamaのmain.goの内容をここに記述）
	// ここではダミーで /api/version だけ返す
	r.GET("/api/version", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"version": "custom-logging"})
	})

	// 他のOllamaのルーティング・処理をここに追加する必要あり

	r.Run(":11434")
}
